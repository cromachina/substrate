import random

import moderngl as mgl
import moderngl_window as mglw
import numpy as np
import pyrr

vert_code = '''
#version 460
in vec2 position;
uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(position, 0.0, 1.0);
}
'''
frag_code = '''
#version 460
precision highp float;

uniform vec4 color;
out vec4 f_color;

void main()
{
    f_color = color;
}
'''

class Agent():
    all_turns = False
    all_walks = False
    all_twirls_in = False
    all_twirls_out = False

    def __init__(self, angle_field:np.ndarray) -> None:
        self.angle_field = angle_field
        self.pos = pyrr.Vector3([0., 0., 0.])
        self.dir = np.nan
        self.angular_vel = 0
        self.turns = False
        self.walks = False
        self.twirls_in = False
        self.twirls_out = False

    def find_start(self):
        self.dir = np.nan
        self.turns = random.random() < 0.2
        self.walks = random.random() < 0.2
        self.twirls_in = random.random() < 0.2
        self.twirls_out = random.random() < 0.2
        self.angular_vel = random.uniform(-0.1, 0.1)

        # Try to find a random location to branch out from.
        for _ in range(1000):
            i = (random.randrange(0, self.angle_field.shape[0]), random.randrange(0, self.angle_field.shape[1]))
            if (not np.isnan(self.angle_field[i])):
                self.pos.xy = i
                flip = -1 if random.random() < 0.5 else 1
                self.dir = self.angle_field[i] + flip * np.pi / 2
                break

    def move(self):
        if np.isnan(self.dir):
            self.find_start()
            return False

        old_pos_index = self.pos.astype(np.int32)
        self.pos = pyrr.Matrix33.from_z_rotation(self.dir) * pyrr.Vector3([1., 0., 0.]) + self.pos

        # Various fun things to change trajectory
        if self.all_walks or self.walks:
            self.angular_vel = random.uniform(-0.3, 0.3)
        if self.all_twirls_in or self.twirls_in:
            self.angular_vel *= 1.001
        if self.all_twirls_out or self.twirls_out:
            self.angular_vel *= 0.999
        if self.all_turns or self.turns:
            self.dir += self.angular_vel

        # Restart when we crash into a path or go out of bounds.
        new_pos_index = self.pos.astype(np.int32)
        if (new_pos_index.x >= 0 and new_pos_index.x < self.angle_field.shape[0] and
            new_pos_index.y >= 0 and new_pos_index.y < self.angle_field.shape[1]):
            if old_pos_index != new_pos_index and not np.isnan(self.angle_field[tuple(new_pos_index.xy)]):
                self.find_start()
                return True
            else:
                self.angle_field[tuple(old_pos_index.xy)] = self.dir
        else:
            self.find_start()
        return False

class Substrate(mglw.WindowConfig):
    gl_version = (4,6)
    title = 'substrate'
    window_size = (1024, 512)
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.identity_mat = pyrr.matrix44.create_identity()
        self.proj_mat = pyrr.matrix44.create_orthogonal_projection(0, self.window_size[0], 0, self.window_size[1], -1, 1)

        # Simulation data
        self.iteration = 0
        self.max_iterations = 1000
        self.max_agents = 100
        self.agent_buffer_data = np.zeros((self.max_agents, 2)).astype(np.float32)
        self.angle_field = np.full(self.window_size, np.nan)
        self.agents:list[Agent] = []

        # GPU program
        self.prog = self.ctx.program(vertex_shader=vert_code, fragment_shader=frag_code)
        # GPU data
        self.agent_buffer = self.ctx.buffer(self.agent_buffer_data)
        self.screen_tri_buffer = self.ctx.buffer(np.array([[-1, -1], [3, -1], [-1, 3]], dtype=np.float32))
        # GPU data organization for shader
        self.agent_vao = self.ctx.vertex_array(self.prog, self.agent_buffer, 'position')
        self.screen_tri_vao = self.ctx.vertex_array(self.prog, self.screen_tri_buffer, 'position')
        # Single buffer so that we can accumulate draws
        self.render_buffer = self.ctx.renderbuffer(self.window_size)
        self.fbo = self.ctx.framebuffer(self.render_buffer)
        self.fbo.clear(1.0, 1.0, 1.0, 1.0)

        self.ctx.point_size = 2
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.DEFAULT_BLENDING

        self.reset_agents()

    def reset_agents(self):
        Agent.all_turns = random.random() < 0.2
        Agent.all_walks = random.random() < 0.2
        Agent.all_twirls_in = random.random() < 0.2
        Agent.all_twirls_out = random.random() < 0.2

        self.angle_field.fill(np.nan)

        flat_field = self.angle_field.ravel()

        for _ in range(16):
            i = random.randrange(0, flat_field.shape[0])
            flat_field[i] = random.random() * np.pi * 2

        self.agents = []
        for _ in range(3):
            self.agents.append(Agent(self.angle_field))

    def move_agents(self):
        self.iteration += 1
        if self.iteration >= self.max_iterations:
            self.iteration = 0
            self.reset_agents()
        new_agents = []
        for agent in self.agents:
            if agent.move() and len(new_agents) + len(self.agents) < self.max_agents:
                new_agents.append(Agent(self.angle_field))
        self.agents.extend(new_agents)

    def draw_agents(self):
        self.prog['color'] = (0., 0., 0., 1.)
        self.prog['projection'].write(self.proj_mat.astype('f4'))
        for i in range(len(self.agents)):
            agent = self.agents[i]
            self.agent_buffer_data[i,:] = agent.pos.xy
        self.agent_buffer.write(self.agent_buffer_data)
        self.agent_vao.render(mgl.POINTS, vertices=len(self.agents))

    def draw_fade_screen(self):
        if self.iteration == 0:
            self.prog['color'] = (1., 1., 1., 0.5)
            self.prog['projection'].write(self.identity_mat.astype('f4'))
            self.screen_tri_vao.render(mgl.TRIANGLES)

    def render(self, time, frametime):
        self.move_agents()
        self.fbo.use()
        self.draw_agents()
        self.draw_fade_screen()
        self.ctx.copy_framebuffer(self.ctx.screen, self.fbo)

if __name__ == '__main__':
    mglw.run_window_config(Substrate)
