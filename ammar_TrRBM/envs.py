from taylor_baseline.lib.env.threedmountain_car import ThreeDMountainCarEnv
from gym import envs

ENVS_DICTIONARY = {
    '3DMountainCar': ThreeDMountainCarEnv
    , '2DMountainCar': envs.make('MountainCar-v0')
    , 'CartPole': envs.make('CartPole-v1')

}
