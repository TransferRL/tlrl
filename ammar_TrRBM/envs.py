from lib.env.threedmountain_car import ThreeDMountainCarEnv
from lib.env.cartpole import CartPoleEnv
from lib.env.threedcartpole import ThreeDCartPoleEnv
from lib.env.mountain_car import MountainCarEnv
from lib.env.acrobot import AcrobotEnv

ENVS_DICTIONARY = {
    '3DMountainCar': ThreeDMountainCarEnv
    , '2DMountainCar': MountainCarEnv
    , '2DCartPole': CartPoleEnv
    , '3DCartPole': ThreeDCartPoleEnv

}
#
ENVS_PATH_DICTIONARY = {
    '3DMountainCar': {'env': ThreeDMountainCarEnv, 'instances_path': '../data/3d_mountain_car/'}
    , '2DMountainCar': {'env': MountainCarEnv, 'instances_path': '../data/2d_mountain_car/'}
    , '2DCartPole': {'env': CartPoleEnv, 'instances_path': '../data/2d_cart_pole/'}
    , '3DCartPole': {'env': ThreeDCartPoleEnv, 'instances_path': '../data/3d_cart_pole/'}
    , 'Acrobot': {'env': AcrobotEnv, 'instances_path': '../data/acrobot/'}
}

