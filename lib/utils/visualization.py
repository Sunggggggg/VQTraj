import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def traj_vis_different_view(traj, color='r', animation_path='test.gif'):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    def init():    
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax1.view_init(azim=-45, elev=45)

        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(-1, 1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        ax2.view_init(azim=-90, elev=90)
    
    def update(frame):
        # ax1.clear()
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax1.view_init(azim=-45, elev=45)
        ax1.dist = 7.5

        # ax2.clear()
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(-1, 1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        ax2.view_init(azim=-90, elev=90)
        ax2.dist = 7.5

        # 현재 프레임의 18개 관절 위치 데이터
        joints = traj[frame]    # [T, 1, 3]
        J = joints.shape[0]

        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        
        ax1.scatter(x, y, z, c=color)
        ax2.scatter(x, y, z, c=color)
    
    length = traj.shape[0]
    interval = 50
    ani = FuncAnimation(fig, update, frames=length, interval=interval, repeat=False, init_func=init)
    ani.save(animation_path, writer=PillowWriter(fps=20))


def traj_vis_different_traj(traj1, traj2, animation_path='test.gif'):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111, projection='3d')
    
    def init():    
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax1.view_init(azim=-45, elev=45)
    
    def update(frame):
        # ax1.clear()
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax1.view_init(azim=-45, elev=45)
        ax1.dist = 7.5

        # 현재 프레임의 18개 관절 위치 데이터
        joints = traj1[frame]    # [T, 1, 3]
        J = joints.shape[0]

        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        ax1.scatter(x, y, z, c='r')

        joints = traj2[frame]    # [T, 1, 3]
        J = joints.shape[0]

        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        ax1.scatter(x, y, z, c='b')
    
    length = traj2.shape[0]
    interval = 50
    ani = FuncAnimation(fig, update, frames=length, interval=interval, repeat=False, init_func=init)
    ani.save(animation_path, writer=PillowWriter(fps=20))