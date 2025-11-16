from typing import List, Dict, Tuple, Set
import numpy as np
from shapely import LineString

from .utils import vector_set_coordinates_to_local_frame

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.utils import get_distance_between_map_object_and_point, extract_polygon_from_map_object
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, SemanticMapLayer
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
MapObjectPolylines, 
VectorFeatureLayer, 
LaneSegmentLaneIDs, 
VectorFeatureLayerMapping, 
LaneSegmentTrafficLightData,
get_traffic_light_encoding,
get_map_object_polygons
)


def interpolate_points(line, num_point, interpolation=None):
    line = LineString(line)
    new_line = np.concatenate([line.interpolate(d).coords._coords for d in np.linspace(0, line.length, num_point)])

    return new_line

def convert_lane_to_fixed_size(ego_pose, feature_coords, speed_limit, lane_route, left_boundary, right_boundary, feature_tl_data, max_elements, max_points,
                                         traffic_light_encoding_dim, interpolation):

    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        raise ValueError(f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}")
    
    lane_has_speed_limit = speed_limit['lane_has_speed_limit']
    lane_speed_limit = speed_limit['lane_speed_limit']

    # trim or zero-pad elements to maintain fixed size
    coords_tensor = np.zeros((max_elements, max_points, 2), dtype=np.float64)
    left_tensor = np.zeros((max_elements, max_points, 2), dtype=np.float64)
    right_tensor = np.zeros((max_elements, max_points, 2), dtype=np.float64)

    lane_has_speed_limit_tensor = np.zeros((max_elements, 1), dtype=np.bool_)
    lane_speed_limit_tensor = np.zeros((max_elements, 1), dtype=np.float32)
    lane_routes = []
        
    avails_tensor = np.zeros((max_elements, max_points), dtype=np.bool_)
    tl_data_tensor = (
        np.zeros((max_elements, max_points, traffic_light_encoding_dim), dtype=np.float32)
        if feature_tl_data is not None else None
    )

    # get elements according to the mean distance to the ego pose, pick the closest 'max_elements' points
    mapping = {}
    for i, e in enumerate(feature_coords):
        dist = np.linalg.norm(e - ego_pose[None, :2], axis=-1).min()
        mapping[i] = dist

    mapping = sorted(mapping.items(), key=lambda item: item[1])
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    for idx, element_idx in enumerate(sorted_elements):
        element_coords = feature_coords[element_idx[0]]
        left_coords = left_boundary[element_idx[0]]
        right_coords = right_boundary[element_idx[0]]

        # interpolate to maintain fixed size if the number of points is not enough
        element_coords = interpolate_points(element_coords, max_points, interpolation=interpolation)
        left_coords = interpolate_points(left_coords, max_points, interpolation=interpolation)
        right_coords = interpolate_points(right_coords, max_points, interpolation=interpolation)

        coords_tensor[idx] = element_coords
        left_tensor[idx] = left_coords
        right_tensor[idx] = right_coords
        avails_tensor[idx] = True  # specify real vs zero-padded data

        lane_has_speed_limit_tensor[idx] = lane_has_speed_limit[element_idx[0]]
        lane_speed_limit_tensor[idx] = lane_speed_limit[element_idx[0]]
        lane_routes.append(lane_route[element_idx[0]])

        if tl_data_tensor is not None and feature_tl_data is not None:
            tl_data_tensor[idx] = feature_tl_data[element_idx[0]]

    return coords_tensor, left_tensor, right_tensor, tl_data_tensor, avails_tensor, lane_has_speed_limit_tensor, lane_speed_limit_tensor, lane_routes



def convert_feature_layer_to_fixed_size(ego_pose, feature_coords, feature_tl_data, max_elements, max_points,
                                         traffic_light_encoding_dim, interpolation):
    """
    Converts variable sized map features to fixed size tensors. Map elements are padded/trimmed to max_elements size.
        Points per feature are interpolated to maintain max_points size.
    :param ego_pose: the current pose of the ego vehicle.
    :param feature_coords: Vector set of coordinates for collection of elements in map layer.
        [num_elements, num_points_in_element (variable size), 2]
    :param feature_tl_data: Optional traffic light status corresponding to map elements at given index in coords.
        [num_elements, traffic_light_encoding_dim (4)]
    :param max_elements: Number of elements to pad/trim to.
    :param max_points: Number of points to interpolate or pad/trim to.
    :param traffic_light_encoding_dim: Dimensionality of traffic light data.
    :param interpolation: Optional interpolation mode for maintaining fixed number of points per element.
        None indicates trimming and zero-padding to take place in lieu of interpolation. Interpolation options: 'linear' and 'area'.
    :return
        coords_tensor: The converted coords tensor.
        tl_data_tensor: The converted traffic light data tensor (if available).
        avails_tensor: Availabilities tensor identifying real vs zero-padded data in coords_tensor and tl_data_tensor.
    :raise ValueError: If coordinates and traffic light data size do not match.
    """
    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        raise ValueError(f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}")

    # trim or zero-pad elements to maintain fixed size
    coords_tensor = np.zeros((max_elements, max_points, 2), dtype=np.float64)
    avails_tensor = np.zeros((max_elements, max_points), dtype=np.bool_)
    tl_data_tensor = (
        np.zeros((max_elements, max_points, traffic_light_encoding_dim), dtype=np.float32)
        if feature_tl_data is not None else None
    )

    # get elements according to the mean distance to the ego pose
    mapping = {}
    for i, e in enumerate(feature_coords):
        dist = np.linalg.norm(e - ego_pose[None, :2], axis=-1).min()
        mapping[i] = dist

    mapping = sorted(mapping.items(), key=lambda item: item[1])
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    for idx, element_idx in enumerate(sorted_elements):
        element_coords = feature_coords[element_idx[0]]
    
        # interpolate to maintain fixed size if the number of points is not enough
        element_coords = interpolate_points(element_coords, max_points, interpolation=interpolation)
        coords_tensor[idx] = element_coords
        avails_tensor[idx] = True  # specify real vs zero-padded data

        if tl_data_tensor is not None and feature_tl_data is not None:
            tl_data_tensor[idx] = feature_tl_data[element_idx[0]]

    return coords_tensor, tl_data_tensor, avails_tensor


def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D, # point is the ego coordinates
    radius: float,
    traffic_light_status_data: List[TrafficLightStatusData],
) -> Tuple[Dict[str, MapObjectPolylines], Dict[str, LaneSegmentTrafficLightData]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param route_roadblock_ids: List of ids of roadblocks/roadblock connectors (lane groups) within goal route.
    :param traffic_light_status_data: A list of all available data at the current time step.
    :return:
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """
    coords: Dict[str, MapObjectPolylines] = {}
    traffic_light_data: Dict[str, LaneSegmentTrafficLightData] = {}
    speed_limit = {}
    feature_layers: List[VectorFeatureLayer] = []

    for feature_name in map_features:
        try:
            if feature_name == 'ROUTE_POLYGON':
                continue
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(f"Object representation for layer: {feature_name} is unavailable")

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids, lane_speed_limit, lane_has_speed_limit, lane_route = get_lane_polylines(map_api, point, radius)

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid
        speed_limit['lane_has_speed_limit'] = np.array(lane_has_speed_limit, dtype=np.bool_)
        speed_limit['lane_speed_limit'] = np.array(lane_speed_limit, dtype=np.float32)
        

        # lane traffic light data
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_status_data
        )

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lanes_left.polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lanes_right.polylines)


    # extract generic map objects
    for feature_layer in feature_layers:
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
            polygons = get_map_object_polygons(
                map_api, point, radius, VectorFeatureLayerMapping.semantic_map_layer(feature_layer)
            )
            coords[feature_layer.name] = polygons

    return coords, traffic_light_data, speed_limit, lane_route


def get_lane_polylines(
    map_api: AbstractMap, point: Point2D, radius: float
) -> Tuple[MapObjectPolylines, MapObjectPolylines, MapObjectPolylines, LaneSegmentLaneIDs]:
    """
    Extract ids, baseline path polylines, and boundary polylines of neighbor lanes and lane connectors around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :return:
        lanes_mid: extracted lane/lane connector baseline polylines.
        lanes_left: extracted lane/lane connector left boundary polylines.
        lanes_right: extracted lane/lane connector right boundary polylines.
        lane_ids: ids of lanes/lane connector associated polylines were extracted from.
    """
    lanes_mid: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_left: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_right: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lane_ids: List[str] = []  # shape: [num_lanes]
    lane_speed_limit = []
    lane_has_speed_limit = []
    lane_roadblock_ids = []
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)

    map_objects = []

    for layer_name in layer_names:
        map_objects += layers[layer_name]
    # sort by distance to query point
    map_objects.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))

    for map_obj in map_objects:
        # center lane
        baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.baseline_path.discrete_path]
        lanes_mid.append(baseline_path_polyline)

        # boundaries
        lanes_left.append([Point2D(node.x, node.y) for node in map_obj.left_boundary.discrete_path])
        lanes_right.append([Point2D(node.x, node.y) for node in map_obj.right_boundary.discrete_path])

        # lane ids
        lane_ids.append(map_obj.id)

        # speed limit
        if map_obj.speed_limit_mps is None:
            lane_speed_limit.append(0.0)
            lane_has_speed_limit.append(False)
        else:
            lane_speed_limit.append(map_obj.speed_limit_mps)
            lane_has_speed_limit.append(True)

        
        lane_roadblock_ids.append(map_obj.get_roadblock_id())

    return (
        MapObjectPolylines(lanes_mid),
        MapObjectPolylines(lanes_left),
        MapObjectPolylines(lanes_right),
        LaneSegmentLaneIDs(lane_ids),
        lane_speed_limit,
        lane_has_speed_limit,
        lane_roadblock_ids
    )

def get_route_polygon_from_roadblock_ids(
    map_api: AbstractMap, route_roadblock_ids: List[str]
) -> MapObjectPolylines:

    route_polygons: List[List[Point2D]] = []


    for route_roadblock_id in route_roadblock_ids:
        # roadblock
        roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK)

        # roadblock connector
        if not roadblock_obj:
            roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)

        if roadblock_obj:
            polygon = extract_polygon_from_map_object(roadblock_obj)
            route_polygons.append(polygon)

    return MapObjectPolylines(route_polygons)


def prune_route_by_connectivity(route_roadblock_ids: List[str], roadblock_ids: Set[str]) -> List[str]:
    """
    Prune route by overlap with extracted roadblock elements within query radius to maintain connectivity in route
    feature. Assumes route_roadblock_ids is ordered and connected to begin with.
    :param route_roadblock_ids: List of roadblock ids representing route.
    :param roadblock_ids: Set of ids of extracted roadblocks within query radius.
    :return: List of pruned roadblock ids (connected and within query radius).
    """
    pruned_route_roadblock_ids: List[str] = []
    route_start = False  # wait for route to come into query radius before declaring broken connection

    for roadblock_id in route_roadblock_ids:

        if roadblock_id in roadblock_ids:
            pruned_route_roadblock_ids.append(roadblock_id)
            route_start = True

        elif route_start:  # connection broken
            break

    return pruned_route_roadblock_ids


def map_process(map_api, route_roadblock_ids, anchor_state, coords, traffic_light_data, speed_limit, lane_route, map_features, max_elements, max_points, interpolation_method):
    """
    This function process the data from the raw vector set map data.
    :param anchor_state: The current state of the ego vehicle.
    :param coords: The input data of the vectorized map coordinates.
    :param traffic_light_data: The input data of the traffic light data.
    :return: dict of the map elements.
    """

    # convert data to tensor list
    anchor_state_tensor = np.array([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=np.float64)
    list_tensor_data = {}

    for feature_name, feature_coords in coords.items():
        list_feature_coords = []

        # Pack coords into tensor list
        
        for element_coords in feature_coords.to_vector():
            list_feature_coords.append(np.array(element_coords, dtype=np.float64) if feature_name != 'STOP_LINE' else np.array(element_coords, dtype=np.float64)[:2])
        list_tensor_data[f"coords.{feature_name}"] = list_feature_coords

        # Pack traffic light data into tensor list if it exists
        if feature_name in traffic_light_data:
            list_feature_tl_data = []

            for element_tl_data in traffic_light_data[feature_name].to_vector():
                list_feature_tl_data.append(np.array(element_tl_data, dtype=np.float64))
            list_tensor_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data

    """
    Vector set map data structure, including:
    coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample.
    traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
    availabilities: Dict[str, List[<np.ndarray: num_elements, num_points>]].
            Boolean indicator of whether feature data is available for point at given index or if it is zero-padded.
    """
    
    tensor_output = {}
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    for feature_name in map_features:
        if f"coords.{feature_name}" in list_tensor_data:
            feature_coords = list_tensor_data[f"coords.{feature_name}"]

            feature_tl_data = (
                list_tensor_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in list_tensor_data
                else None
            )

            if feature_name == 'LANE':
                coords, left_coords, right_coords, tl_data, avails, lane_has_speed_limit_tensor, lane_speed_limit_tensor, lane_routes = convert_lane_to_fixed_size(
                        anchor_state_tensor,
                        feature_coords,
                        speed_limit,
                        lane_route,
                        list_tensor_data[f"coords.LEFT_BOUNDARY"],
                        list_tensor_data[f"coords.RIGHT_BOUNDARY"],
                        feature_tl_data,
                        max_elements[feature_name],
                        max_points[feature_name],
                        traffic_light_encoding_dim,
                        interpolation=interpolation_method  # apply interpolation only for lane features
                        if feature_name
                        in [
                            VectorFeatureLayer.LANE.name,
                        ]
                        else None,
                )
                left_coords = vector_set_coordinates_to_local_frame(left_coords, avails, anchor_state_tensor)
                right_coords = vector_set_coordinates_to_local_frame(right_coords, avails, anchor_state_tensor)
                tensor_output[f"vector_set_map.coords.LEFT_BOUNDARY"] = left_coords
                tensor_output[f"vector_set_map.coords.RIGHT_BOUNDARY"] = right_coords

                '''
                Get roadblock polygon
                '''
                lane_on_route = []
                pruned_lane_roadblock_ids = [route for route in route_roadblock_ids if route in lane_routes]
                pruned_route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids, pruned_lane_roadblock_ids)

                for route in lane_routes:
                    lane_on_route.append(route in pruned_route_roadblock_ids)

                route_polygon = get_route_polygon_from_roadblock_ids(map_api, pruned_route_roadblock_ids)

                # Pack coords into tensor list
                list_feature_coords = []
                for element_coords in route_polygon.to_vector():
                    list_feature_coords.append(np.array(element_coords, dtype=np.float64))
                list_tensor_data[f"coords.ROUTE_POLYGON"] = list_feature_coords


            elif feature_name == 'LEFT_BOUNDARY' or feature_name == 'RIGHT_BOUNDARY':
                continue
            else:
                coords, tl_data, avails = convert_feature_layer_to_fixed_size(
                        anchor_state_tensor,
                        feature_coords,
                        feature_tl_data,
                        max_elements[feature_name],
                        max_points[feature_name],
                        traffic_light_encoding_dim,
                        interpolation=interpolation_method  # apply interpolation only for lane features
                        if feature_name
                        in [
                            VectorFeatureLayer.CROSSWALK.name,
                            VectorFeatureLayer.STOP_LINE.name,
                            "ROUTE_POLYGON"
                        ]
                        else None,
                )

            coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state_tensor)

            tensor_output[f"vector_set_map.coords.{feature_name}"] = coords
            tensor_output[f"vector_set_map.availabilities.{feature_name}"] = avails

            if tl_data is not None:
                tensor_output[f"vector_set_map.traffic_light_data.{feature_name}"] = tl_data


    """
    Post-precoss the map elements to different map types. Each map type is a array with the following shape.
    N: number of map elements (fixed for a given map feature)
    P: number of points (fixed for a given map feature)
    F: number of features
    """

    for feature_name in map_features:
        if feature_name == "LANE":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}']
            left_boundary = tensor_output[f"vector_set_map.coords.LEFT_BOUNDARY"]
            right_boundary = tensor_output[f"vector_set_map.coords.RIGHT_BOUNDARY"]
            traffic_light_state = tensor_output[f'vector_set_map.traffic_light_data.{feature_name}']
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}']
            vector_map_lanes = lane_polyline_process(polylines, left_boundary, right_boundary, avails, traffic_light_state)

        elif feature_name == "CROSSWALK":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}']
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}']
            vector_map_crosswalks = polyline_process(polylines, avails)

        elif feature_name == "STOP_LINE":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}']
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}']
            vector_map_stopline = polyline_process(polylines, avails)

        elif feature_name == "ROUTE_POLYGON":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}']
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}']
            vector_map_route_polygons = polyline_process(polylines, avails)

        elif feature_name == "ROUTE_LANES":
            loc = 0
            # TODO: add has speed limit
            vector_map_route_lanes = np.zeros((max_elements["ROUTE_LANES"], vector_map_lanes.shape[-2], vector_map_lanes.shape[-1]), dtype=np.float32)
            route_lanes_speed_limit = np.zeros((max_elements["ROUTE_LANES"], 1), dtype=np.float32)
            route_lanes_has_speed_limit = np.zeros((max_elements["ROUTE_LANES"], 1), dtype=np.bool_)
            for i in range(len(lane_on_route)):
                if lane_on_route[i] == True:
                    vector_map_route_lanes[loc] = vector_map_lanes[i]
                    route_lanes_speed_limit[loc] = lane_speed_limit_tensor[i]
                    route_lanes_has_speed_limit[loc] = lane_has_speed_limit_tensor[i]
                    loc += 1
                if loc == max_elements["ROUTE_LANES"]:
                    break

        else:
            pass

    vector_map_output = {'lanes': vector_map_lanes, 'lanes_speed_limit': lane_speed_limit_tensor, 'lanes_has_speed_limit': lane_has_speed_limit_tensor, \
                            'route_lanes': vector_map_route_lanes, 'route_polygons': vector_map_route_polygons, \
                            'route_lanes_speed_limit': route_lanes_speed_limit, 'route_lanes_has_speed_limit': route_lanes_has_speed_limit}
    if 'CROSSWALK' in map_features:
        vector_map_output.update({'crosswalks': vector_map_crosswalks})

    return vector_map_output


def polyline_process(polylines, avails):
    dim = 4
    new_polylines = np.zeros(shape=(polylines.shape[0], polylines.shape[1], dim), dtype=np.float32)

    for i in range(polylines.shape[0]):
        if avails[i][0]: 
            polyline = polylines[i]
            polyline_vector = polyline[1:]-polyline[:-1]
            polyline_vector = np.insert(polyline_vector, polyline_vector.shape[0] , 0, axis=0)

            new_polylines[i] = np.concatenate([polyline, polyline_vector], axis=-1)

    return new_polylines

def lane_polyline_process(polylines, left_boundary, right_boundary, avails, traffic_light):
    dim = 12
    new_polylines = np.zeros(shape=(polylines.shape[0], polylines.shape[1], dim), dtype=np.float32) # (num_lanes, num_points, dim)

    # make sure the boundary and the center line has the same direction
    for i in range(polylines.shape[0]):
        if avails[i][0]: 
            polyline = polylines[i]
            polyline_vector = polyline[1:]-polyline[:-1]
            polyline_vector = np.insert(polyline_vector, polyline_vector.shape[0] , 0, axis=0)

            if np.linalg.norm(left_boundary[i, -1] - polyline[0]) < np.linalg.norm(left_boundary[i, 0] - polyline[0]):
                left_boundary[i] = np.flip(left_boundary[i], axis=0)

            if np.linalg.norm(right_boundary[i, -1] - polyline[0]) < np.linalg.norm(right_boundary[i, 0] - polyline[0]):
                right_boundary[i] = np.flip(right_boundary[i], axis=0)

            polyline_to_left = left_boundary[i] - polyline
            polyline_to_right = right_boundary[i] - polyline


            new_polylines[i] = np.concatenate([polyline, polyline_vector, polyline_to_left, polyline_to_right, traffic_light[i]], axis=-1)  

    return new_polylines
