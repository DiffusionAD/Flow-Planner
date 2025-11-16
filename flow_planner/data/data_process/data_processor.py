from typing import List
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import matplotlib.pyplot as plt

from .roadblock_utils import route_roadblock_correction
from .agent_process import (
agent_past_process, 
agent_future_process,
sampled_tracked_objects_to_array_list,
sampled_static_objects_to_array_list
)
from .map_process import get_neighbor_vector_set_map, map_process
from .utils import convert_to_model_inputs

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint, Point2D
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses


DEBUG = False
_SPEED_LIMIT = 0.6 # speed limit used in waymax

class DataProcessor(object):
    def __init__(self, save_dir):
        self._save_dir = save_dir

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = 32
        self.num_static = 5
        self.max_ped_bike = 10

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES', 'ROUTE_POLYGON', 'CROSSWALK'] # name of map features to be extracted.
        self._max_elements = {'LANE': 70, 'LEFT_BOUNDARY': 70, 'RIGHT_BOUNDARY': 70, 'ROUTE_LANES': 25, 'ROUTE_POLYGON': 5, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 20, 'LEFT_BOUNDARY': 20, 'RIGHT_BOUNDARY': 20, 'ROUTE_LANES': 20, 'ROUTE_POLYGON': 10, 'CROSSWALK': 10} # maximum number of points per feature to extract per feature layer.
        self._vehicle_parameters = get_pacifica_parameters()

        self._radius = 100 # [m] query radius scope relative to the current pose. IS IT TOO BIG?
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.

    def sampled_past_ego_states_to_array(self, past_ego_states: List[EgoState]) -> npt.NDArray[np.float32]:

        output = np.zeros((len(past_ego_states), 13), dtype=np.float64)
        for i in range(0, len(past_ego_states), 1):
            
            output[i, EgoInternalIndex.x()] = past_ego_states[i].rear_axle.x
            output[i, EgoInternalIndex.y()] = past_ego_states[i].rear_axle.y
            output[i, EgoInternalIndex.heading()] = past_ego_states[i].rear_axle.heading
            output[i, EgoInternalIndex.vx()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.x
            output[i, EgoInternalIndex.vy()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.y
            output[i, EgoInternalIndex.ax()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.x
            output[i, EgoInternalIndex.ay()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.y
            output[i, 7] = self._vehicle_parameters.width
            output[i, 8] = self._vehicle_parameters.length
            
        output[:, 9] = 1 # ego_type

        return output

    def sampled_past_timestamps_to_array(self, past_time_stamps: List[TimePoint]) -> npt.NDArray[np.float32]:
        flat = [t.time_us for t in past_time_stamps]
        return np.array(flat, dtype=np.int64)

    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )


        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state] # the sampled_ego_states includes past and current
        past_ego_states_array = self.sampled_past_ego_states_to_array(sampled_past_ego_states)

        
        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_array = self.sampled_past_timestamps_to_array(past_time_stamps)

        return past_ego_states_array, past_time_stamps_array
    
    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_array_list, past_tracked_objects_types = \
              sampled_tracked_objects_to_array_list(sampled_past_observations)
        
        current_static_objects_array_list, current_static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

        return past_tracked_objects_array_list, past_tracked_objects_types, current_static_objects_array_list, current_static_objects_types

    def get_map(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)

        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = list(self.scenario.get_traffic_light_status_at_iteration(0))

        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, self.map_api, route_roadblock_ids
            )

        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            self.map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )

        # roadblock and map data process
        vector_map = map_process(self.map_api, route_roadblock_ids, ego_state.rear_axle, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        return vector_map

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
        agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, self.num_agents, agent_index)

        return agent_futures
    
    def plot_scenario(self, data, data_type):
        # Create map layers
        create_map_raster(data['lanes'], data['route_lanes'])

        # Create agent layers
        create_ego_raster(data['ego_agent_past'][-1], self._vehicle_parameters)
        # create_agents_raster(data['neighbor_agents_past'][:, -1])

        # Draw past and future trajectories
        draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'])
        draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'])

        plt.gca().set_aspect('equal')
        plt.tight_layout()
        import time
        plt.savefig(f'./img/image_{hash(time.time())}.png', dpi=600)
        plt.clf()

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def calculate_additional_ego_states(self, ego_agent_past, time_stamp):
        # transform haeding to cos h, sin h and calculate the steering_angle and yaw_rate for current state

        current_state = ego_agent_past[-1]
        prev_state = ego_agent_past[-2]

        dt = (time_stamp[-1] - time_stamp[-2]) * 1e-6

        cur_velocity = current_state[3]
        angle_diff = current_state[2] - prev_state[2]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = angle_diff / dt

        if abs(cur_velocity) < 0.2:
            steering_angle = 0.0
            yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable
        else:
            steering_angle = np.arctan(
                yaw_rate * get_pacifica_parameters().wheel_base / abs(cur_velocity)
            )
            steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
            yaw_rate = np.clip(yaw_rate, -0.95, 0.95)


        past = np.zeros((ego_agent_past.shape[0], ego_agent_past.shape[1]+1), dtype=np.float32)

        past[:, :2] = ego_agent_past[:, :2]
        past[:, 2] = np.cos(ego_agent_past[:, 2])
        past[:, 3] = np.sin(ego_agent_past[:, 2])
        past[:, 4:] = ego_agent_past[:, 3:]

        current = np.zeros((ego_agent_past.shape[1]+3), dtype=np.float32)
        current[:2] = current_state[:2]
        current[2] = np.cos(current_state[2])
        current[3] = np.sin(current_state[2])
        current[4:8] = current_state[3:7]
        current[8] = steering_angle
        current[9] = yaw_rate
        current[10:] = current_state[7:]

        return past, current

    def work(self, scenarios):
        self.i = 0
        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            self.scenario = scenario
            self.map_api = scenario.map_api        

            # get agent past tracks
            ego_agent_past, time_stamps_past = self.get_ego_agent()

            neighbor_agents_past, neighbor_agents_types, static_objects, static_objects_types = self.get_neighbor_agents()

            ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
                agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike)
            
            # get vector set map
            vector_map = self.get_map()

            # # get agent future tracks
            ego_agent_future = self.get_ego_agent_future()
            neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)


            ego_agent_past, ego_current_state = self.calculate_additional_ego_states(ego_agent_past, time_stamps_past)

            # # gather data
            data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
            data.update(vector_map)

            self.save_to_disk(self._save_dir, data)


    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''
        ego_state_buffer = history_buffer.ego_state_buffer # Past ego state including the current
        ego_agent_past = self.sampled_past_ego_states_to_array(ego_state_buffer)
        time_stamps_past = self.sampled_past_timestamps_to_array([state.time_point for state in ego_state_buffer])
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)

        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike)
    

        '''
        Map
        '''
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(map_api, route_roadblock_ids, ego_state.rear_axle, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points, None)

        ego_agent_past = ego_agent_past[-21:]
        ego_agent_past, ego_current_state = self.calculate_additional_ego_states(ego_agent_past, time_stamps_past)
        
        data = {"ego_agent_past": ego_agent_past, 
                "neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": ego_current_state,
                "static_objects": static_objects}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data