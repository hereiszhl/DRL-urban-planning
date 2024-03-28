import copy
from typing import Dict, Tuple, Text, List, Optional


import libpysal
import math
import momepy
import networkx as nx
import pandas as pd
from geopandas import GeoSeries, GeoDataFrame
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point, MultiPolygon, MultiPoint, LineString, MultiLineString
from shapely.ops import snap, polygonize

from urban_planning.envs import city_config
from khrylib.utils import load_yaml, load_pickle, simplify_by_angle, simplify_by_distance, get_boundary_edges, \
    slice_polygon_from_edge, slice_polygon_from_corner, get_intersection_polygon_with_maximum_area
from khrylib.utils import set_land_use_array_from_dict


class PlanClient(object):
    """Defines the PlanClient class."""
    PLAN_ORDER = np.array([  #确定规划元素的处理优先级
        city_config.HOSPITAL_L,
        city_config.SCHOOL,
        city_config.HOSPITAL_S,
        city_config.RECREATION,
        city_config.RESIDENTIAL,
        city_config.GREEN_L,
        city_config.OFFICE,
        city_config.BUSINESS,
        city_config.GREEN_S], dtype=np.int32)
    EPSILON = 1E-4  # 用于浮点数比较的小数
    DEG_TOL = 1   # 角度容差
    SNAP_EPSILON = 1     # 对齐操作的容差
### EPSILON、DEG_TOL 和 SNAP_EPSILON 是类属性，它们定义了一些数值常量，可能用于计算时的容差或精度控制
    def __init__(self, objectives_plan_file: Text, init_plan_file: Text) -> None:
        """Creates a PlanClient client object.

        Args:
            objectives_plan_file: Path to the file of community objectives.
            init_plan_file: Path to the file of initial plan.
        """
        # 构造函数，用于创建 PlanClient 对象，并加载规划目标和初始规划文件
        file_path = 'urban_planning/cfg/**/{}.yaml'.format(objectives_plan_file)  #输入规划目标的文件位置
        self.objectives = load_yaml(file_path)
        file_path = 'urban_planning/cfg/**/{}.pickle'.format(init_plan_file)   #输入原始土地情况文件的位置
        self.init_plan = load_pickle(file_path)
        self.init_objectives()
        self.init_constraints()
        self.restore_plan()

    def init_objectives(self) -> None:
        """Initializes objectives of different land uses."""
        objectives = self.objectives   # 从 self.objectives 属性中获取社区和土地用途的目标
        self._grid_cols = objectives['community']['grid_cols']    # 初始化社区网格的列数、行数、单元格边长和面积
        self._grid_rows = objectives['community']['grid_rows']
        self._cell_edge_length = objectives['community']['cell_edge_length']
        self._cell_area = self._cell_edge_length ** 2   # 计算单元格面积

        # 获取计划中包含的土地用途类型，并将它们转换为对应的 ID
        land_use_types_to_plan = objectives['objectives']['land_use']    
        land_use_to_plan = np.array(
            [city_config.LAND_USE_ID_MAP[land_use] for land_use in land_use_types_to_plan],
            dtype=np.int32)
        custom_planning_order = objectives['objectives'].get('custom_planning_order', False)
        if custom_planning_order:
            self._plan_order = land_use_to_plan
        else:
            self._plan_order = self.PLAN_ORDER[np.isin(self.PLAN_ORDER, land_use_to_plan)]

        self._required_plan_ratio = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_plan_ratio = objectives['objectives']['ratio']
        set_land_use_array_from_dict(self._required_plan_ratio, required_plan_ratio, city_config.LAND_USE_ID_MAP)

        self._required_plan_count = np.zeros(city_config.NUM_TYPES, dtype=np.int32)
        required_plan_count = objectives['objectives']['count']
        set_land_use_array_from_dict(
            self._required_plan_count, required_plan_count, city_config.LAND_USE_ID_MAP)

    def init_constraints(self) -> None:  # 用于初始化不同土地使用的约束条件
        """Initializes constraints of different land uses."""
        objectives = self.objectives
        self.init_specific_constraints(objectives['constraints'])
        self.init_common_constraints()

    
    def init_specific_constraints(self, constraints: Dict) -> None:    #导入适用于特定土地类型的规划约束条件
        """Initializes constraints of specific land uses.

        Args:
            constraints: Constraints of specific land uses.
        """

        # 初始化四个NumPy数组来存储不同土地类型的最大面积、最小面积、最大边长和最小边长的约束。
        # 确保土地使用规划满足特定的空间约束，如面积和边长。
        self._required_max_area = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_max_area = constraints['max_area']
        set_land_use_array_from_dict(self._required_max_area, required_max_area, city_config.LAND_USE_ID_MAP)

        # set_land_use_array_from_dict 函数用于根据 city_config.LAND_USE_ID_MAP 映射，从 constraints 字典中提取相应的值并设置到数组中。
        self._required_min_area = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_min_area = constraints['min_area']
        set_land_use_array_from_dict(self._required_min_area, required_min_area, city_config.LAND_USE_ID_MAP)

        self._required_max_edge_length = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_max_edge_length = constraints['max_edge_length']
        set_land_use_array_from_dict(
            self._required_max_edge_length, required_max_edge_length, city_config.LAND_USE_ID_MAP)

        self._required_min_edge_length = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        required_min_edge_length = constraints['min_edge_length']
        set_land_use_array_from_dict(
            self._required_min_edge_length, required_min_edge_length, city_config.LAND_USE_ID_MAP)

    def init_common_constraints(self) -> None:    # 导入适用于所有土地类型的约束条件
        """Initializes common constraints of difference land uses."""
        self._common_max_area = self._required_max_area[self._plan_order].max()
        self._common_min_area = self._required_min_area[self._plan_order].min()
        self._common_max_edge_length = self._required_max_edge_length[self._plan_order].max()
        self._common_min_edge_length = self._required_min_edge_length[self._plan_order].min()
        self._min_edge_grid = round(self._common_min_edge_length / self._cell_edge_length)
        self._max_edge_grid = round(self._common_max_edge_length / self._cell_edge_length)

    def get_common_max_area(self) -> float:  #返回所有土地使用类型的共同最大面积
        """Returns the required maximum area of all land uses."""
        return self._common_max_area

    def get_common_max_edge_length(self) -> float:  #返回所有土地使用类型的共同最大边长
        """Returns the required maximum edge length of all land uses."""
        return self._common_max_edge_length

    def _add_domain_features(self) -> None:  #向地理数据框架（gdf）添加几何特征，如矩形度、等效矩形指数和正方形紧凑度
        """Adds domain features to the gdf."""
        self._gdf['rect'] = momepy.Rectangularity(self._gdf[self._gdf.geom_type == 'Polygon']).series
        self._gdf['eqi'] = momepy.EquivalentRectangularIndex(self._gdf[self._gdf.geom_type == 'Polygon']).series
        self._gdf['sc'] = momepy.SquareCompactness(self._gdf[self._gdf.geom_type == 'Polygon']).series

    def get_init_plan(self) -> Dict:   #获取初始的规划方案
        """Returns the initial plan."""
        return self.init_plan

    def restore_plan(self) -> None:   #恢复初始的规划方案
        """Restore the initial plan."""
        self._initial_gdf = self.init_plan['gdf']
        self._gdf = copy.deepcopy(self._initial_gdf)
        self._add_domain_features()
        self._load_concept(self.init_plan.get('concept', list()))
        self._rule_constraints = self.init_plan.get('rule_constraints', False)
        self._init_stats()
        self._init_counter()

    def load_plan(self, gdf: GeoDataFrame) -> None:   #加载一个新的规划方案
        """Loads the given plan.

        Args:
            gdf: The plan to load.
        """
        self._gdf = copy.deepcopy(gdf)

    def _load_concept(self, concept: List) -> None:    #初始化规划方案的概念
        """Initializes the planning concept of the plan.

        Args:
            concept: The planning concept.
        """
        self._concept = concept

    def _init_stats(self) -> None:    #初始化规划方案的各项数据，如不同土地使用类型的面积，比例和数量等。
        """Initialize statistics of the plan."""
        #确定所需要增加或减少的规划面积。首先计算存在的地块的总面积和外部地块的面积，
        #然后计算社区区域的面积（总面积减去外部面积）。接着，它根据社区区域的面积和预定的规划面积比例计算所需的规划面积。
        gdf = self._gdf[self._gdf['existence'] == True]
        total_area = gdf.area.sum()*self._cell_area
        outside_area = gdf[gdf['type'] == city_config.OUTSIDE].area.sum()*self._cell_area
        self._community_area = total_area - outside_area

        self._required_plan_area = self._community_area * self._required_plan_ratio
        self._plan_area = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        self._plan_ratio = np.zeros(city_config.NUM_TYPES, dtype=np.float32)
        self._plan_count = np.zeros(city_config.NUM_TYPES, dtype=np.int32)
        self._compute_stats()

    def _compute_stats(self) -> None:   #更新规划数据
        """Update statistics of the plan."""
        gdf = self._gdf[self._gdf['existence'] == True]
        #遍历所有存在的土地使用类型，并计算每种类型的面积、面积占社区区域的比例以及该类型地块的数量
        for land_use in city_config.LAND_USE_ID:
            area = gdf[gdf['type'] == land_use].area.sum() * self._cell_area
            self._plan_area[land_use] = area
            self._plan_ratio[land_use] = area / self._community_area
            self._plan_count[land_use] = len(gdf[gdf['type'] == land_use])

    def _update_stats(self, land_use_type: int, land_use_area: float) -> None:  #更新给定新土地使用类型的规划统计数据
        """Update statistics of the plan given new land_use.

        Args:
            land_use_type: land use type of the new land use.
            land_use_area: area of the new land use.
        """
        #接受土地使用类型 land_use_type 和新土地使用的面积 land_use_area 作为参数
        #方法增加相应土地使用类型的计数器，更新该类型的总面积，并重新计算该类型占社区总面积的比例。
        #同时，它会从可行土地使用类型的总面积中减去新土地使用的面积，并更新可行土地使用的比例。
        self._plan_count[land_use_type] += 1

        self._plan_area[land_use_type] += land_use_area
        self._plan_ratio[land_use_type] = self._plan_area[land_use_type]/self._community_area

        self._plan_area[city_config.FEASIBLE] -= land_use_area
        self._plan_ratio[city_config.FEASIBLE] = self._plan_area[city_config.FEASIBLE]/self._community_area

    def _init_counter(self):
        """Initialize action ID counter."""
        self._action_id = self._gdf.index.max()

    def _counter(self):
        """Return counter and add one."""
        self._action_id += 1
        return self._action_id

    def unplan_all_land_use(self) -> None:
        """Unplan all land use"""
        self._gdf = copy.deepcopy(self._initial_gdf)
        self._add_domain_features()
        self._compute_stats()
        self._init_counter()

    def freeze_land_use(self, land_use_gdf: GeoDataFrame) -> None:
        """Freeze the given land use.

        Args:
            land_use_gdf: The land use to freeze.
        """
        self._initial_gdf = copy.deepcopy(land_use_gdf)

    def fill_leftover(self) -> None:
        """Fill leftover space."""
        self._gdf.loc[(self._gdf['type'] == city_config.FEASIBLE) & (self._gdf['existence'] == True),
                      'type'] = city_config.GREEN_S  #把畸零地块设置为绿地

    def snapshot(self):   # 创建当前地理数据框架（_gdf）的快照。这通常用于保存当前状态，以便可以在需要时恢复
        """Snapshot the gdf."""
        snapshot = copy.deepcopy(self._gdf)
        return snapshot

    def build_all_road(self):   #将所有标记为边界（city_config.BOUNDARY）且存在的地块的类型更改为道路
        """Build all road"""
        self._gdf.loc[(self._gdf['type'] == city_config.BOUNDARY) & (self._gdf['existence'] == True),
                      'type'] = city_config.ROAD

    #检查土地使用规划是否完成。它通过比较实际的土地使用比例（_plan_ratio）和所需的土地使用比例（_required_plan_ratio），
    #以及土地使用类型的计数（_plan_count）和所需的计数（_required_plan_count）来确定规划是否满足所有条件。
    def is_land_use_done(self) -> bool:    
        """Check if the land_use planning is done."""
        ratio_satisfication = (self._plan_ratio - self._required_plan_ratio >= -self.EPSILON)[self._plan_order].all()
        count_satisfication = (self._plan_count >= self._required_plan_count)[self._plan_order].all()
        done = ratio_satisfication and count_satisfication
        return done

    def get_gdf(self) -> GeoDataFrame:
        """Return the current GDF."""
        return self._gdf

    def _get_current_gdf_and_graph(self) -> Tuple[GeoDataFrame, nx.Graph]:
        """Return the current GDF and graph.

        Returns:
            gdf: current GDF.
            graph: current graph.
                   Nodes are land_use, road intersections and road segments. Edges are spatial contiguity.
        """
        gdf = copy.deepcopy(self._gdf[self._gdf['existence'] == True])
        w = libpysal.weights.fuzzy_contiguity(gdf)
        graph = w.to_networkx()
        self._current_gdf = gdf
        self._current_graph = graph
        return gdf, graph

    #转译规划法规
    def _filter_block_by_rule(self,
                              gdf: GeoDataFrame, feasible_blocks_id: np.ndarray, land_use_type: int) -> np.ndarray:
        """Filter feasible blocks by rule.

        Args:
            gdf: current GDF.
            feasible_blocks_id: feasible blocks ID.
            land_use_type: land use type.

        Returns:
            filtered_blocks_id: filtered blocks.
        """
        # 如果土地使用类型是学校（city_config.SCHOOL），方法会找出所有类型为大型医院（city_config.HOSPITAL_L）的地块，
        # 并获取与这些医院相交的地块的ID。然后，它会从可行地块ID中排除这些靠近大型医院的地块。                          
        if land_use_type == city_config.SCHOOL:
            hospital_l = gdf[gdf['type'] == city_config.HOSPITAL_L].unary_union
            near_hospital_l = gdf[(gdf.geom_type == 'Polygon') & (gdf.intersects(hospital_l))].index.to_numpy()
            filtered_blocks_id = np.setdiff1d(feasible_blocks_id, near_hospital_l)
        #如果土地使用类型是小型医院（city_config.HOSPITAL_S），方法会找出所有类型为学校、大型医院或小型医院的地块，
        #并获取与这些地块相交的地块的ID。然后，它会从可行地块ID中排除这些靠近学校或医院的地块                          
        elif land_use_type == city_config.HOSPITAL_S:
            school = gdf[(gdf['type'] == city_config.SCHOOL) | (gdf['type'] == city_config.HOSPITAL_L) | (gdf['type'] == city_config.HOSPITAL_S)].unary_union
            near_school = gdf[(gdf.geom_type == 'Polygon') & (gdf.intersects(school))].index.to_numpy()
            filtered_blocks_id = np.setdiff1d(feasible_blocks_id, near_school)
        #如果土地使用类型不是学校或小型医院，方法将不会过滤任何地块，
        else:
            filtered_blocks_id = feasible_blocks_id
        return filtered_blocks_id

    def _get_graph_edge_mask(self, land_use_type: int) -> np.ndarray:  #选择一条边进行土地利用规划
        """Return the edge mask of the graph.

        Args:
            land_use_type: land use type of the new land use.

        Returns:
            edge_mask: edge mask of the graph.
        """
        gdf, graph = self._get_current_gdf_and_graph() # 调用 _get_current_gdf_and_graph 方法获取当前的地理数据框架（GDF）和图（Graph）。
        current_graph_edges = np.array(graph.edges)  # 从图中提取所有边缘，并将它们转换为NumPy数组
        current_graph_nodes_id = gdf.index.to_numpy()  # 获取GDF中所有节点的索引，并将这些索引与边缘数组相匹配，创建一个包含边缘和对应节点ID的数组
        self._current_graph_edges_with_id = current_graph_nodes_id[current_graph_edges]

        #筛选出符合以下条件的地块ID：
        feasible_blocks_id = gdf[
            (gdf['type'] == city_config.FEASIBLE) & #地块类型为可行（city_config.FEASIBLE）
            (gdf.area * self._cell_area >= self._required_min_area[land_use_type])].index.to_numpy()  #地块面积乘以单元格面积大于或等于给定土地使用类型所需的最小面积。
        intersections_id = gdf[gdf.geom_type == 'Point'].index.to_numpy()   # 获取所有几何类型为点（即交叉点）的地块ID

        if self._rule_constraints:  #如果有相应的规则约束，则用约束规则进一步筛选可用的地块ID
            feasible_blocks_id = self._filter_block_by_rule(gdf, feasible_blocks_id, land_use_type)

        edge_mask = np.logical_or( #创建一个逻辑掩码 edge_mask，该掩码标识出那些连接可行地块和交叉点的边缘
            np.logical_and(
                np.isin(self._current_graph_edges_with_id[:, 0], feasible_blocks_id),
                np.isin(self._current_graph_edges_with_id[:, 1], intersections_id)
            ),
            np.logical_and(
                np.isin(self._current_graph_edges_with_id[:, 1], feasible_blocks_id),
                np.isin(self._current_graph_edges_with_id[:, 0], intersections_id)
            )
        )

        return edge_mask  # 返回这个逻辑掩码 edge_mask

    def get_current_land_use_and_mask(self) -> Tuple[Dict, np.ndarray]:  #获取当前地块的土地使用类型和掩码
        """Return the current land use and mask.

        Returns:
            land_use: current land use.
            mask: current mask.
        """
        land_use = dict()  #创建一个空字典 land_use 来存储土地使用信息
        ###对于每种土地利用类型，计算剩余的规划面积和剩余的规划数量，这两个值是根据所需的规划面积和数量减去已规划的面积和数量得到的
        remaining_plan_area = (self._required_plan_area - self._plan_area)[self._plan_order] #计算剩余的规划面积
        remaining_plan_count = (self._required_plan_count - self._plan_count)[self._plan_order]  #计算剩余的规划数量
        ###根据剩余的规划面积和规划数量确定土地利用类型
        #代码会检查哪些土地使用类型的剩余规划面积或数量大于一个非常小的正数 (self.EPSILON)。这个正数用来处理浮点数的精度问题，
        #确保即使是非常小的剩余值也能被识别为有效的需求。系统会检查哪些类型的剩余规划面积或数量大于一个非常小的正数（self.EPSILON），
        #这意味着这些类型还没有达到规划目标。系统会从满足上述条件的类型中选择第一个类型作为 land_use_type。
        land_use_type = self._plan_order[np.logical_or(remaining_plan_area > self.EPSILON, remaining_plan_count > 0)][0]  
        land_use['type'] = land_use_type
        mask = self._get_graph_edge_mask(land_use_type)
        #在 land_use 字典中设置土地使用类型和其他相关属性，如坐标、面积、长度、宽度、高度以及几何特征
        #（矩形度 rect、等效矩形指数 eqi、正方形紧凑度 sc）。
        land_use['x'] = 0.5
        land_use['y'] = 0.5
        land_use['area'] = self._required_max_area[land_use_type]
        land_use['length'] = 4*self._required_max_edge_length[land_use_type]
        land_use['width'] = self._required_max_edge_length[land_use_type]
        land_use['height'] = self._required_max_edge_length[land_use_type]
        land_use['rect'] = 1.0
        land_use['eqi'] = 1.0
        land_use['sc'] = 1.0
        return land_use, mask

    def get_current_road_mask(self) -> np.ndarray:   #根据类型为boundary的边缘识别出哪些边缘属于当前现有道路
        """Return the current road mask.

        Returns:
            mask: current road mask.
        """
        gdf, graph = self._get_current_gdf_and_graph()  #调用 _get_current_gdf_and_graph 方法来获取当前的地理数据框架（GDF）和图（Graph）。
        self._current_graph_nodes_id = current_graph_nodes_id = gdf.index.to_numpy()  #从GDF中获取所有节点的索引，并将其存储在 self._current_graph_nodes_id 中
        boundary_id = gdf[gdf['type'] == city_config.BOUNDARY].index.to_numpy()  #找出所有类型为边界（city_config.BOUNDARY）的地块的索引，这些通常代表道路的边缘或边界
        mask = np.isin(current_graph_nodes_id, boundary_id)  #使用 np.isin 函数创建一个掩码 mask，该掩码标识出哪些节点是边界节点


        return mask

    #简化多边形的形状。简化过程可以减少多边形的顶点数量，这对于地图绘制和空间分析非常有用，因为它可以减少计算量并提高处理效率。
    #简化后的多边形仍然保留了原始多边形的基本形状和特征，但以更少的数据表示。
    def _simplify_polygon(self,
                          polygon: Polygon,
                          intersection: Point) -> Tuple[Polygon, GeoSeries, Text, List, float]:
        """Simplify polygon.

        Args:
            polygon: polygon to simplify.
            intersection: intersection point.

        Returns:
            polygon: simplified polygon.
            polygon_boundary: GeoSeries of boundary edges of the simplified polygon.
            relation: relation between the simplified polygon and the intersection point.
            edges: list of boundary edges of the simplified polygon that intersects with the intersection point.
        """
        cached_polygon = polygon
        polygon = simplify_by_angle(polygon.normalize(), deg_tol=self.DEG_TOL)
        simple_coords = MultiPoint(list(polygon.exterior.coords))
        polygon_boundary = get_boundary_edges(polygon, 'GeoSeries')

        error_msg = 'Original polygon: {}'.format(cached_polygon)
        error_msg += '\nSimplified polygon: {}'.format(polygon)
        error_msg += '\nIntersection: {}'.format(intersection)

        if simple_coords.distance(intersection) > self.EPSILON:
            boundary_distance = polygon_boundary.distance(intersection)
            distance = boundary_distance.min()
            if (boundary_distance < distance + self.EPSILON).sum() > 1:
                raise ValueError(error_msg + '\nIntersection within edge is near two edges.')
            relation = 'edge'
            edges = polygon_boundary[boundary_distance < distance + self.EPSILON].to_list()
        elif simple_coords.contains(intersection):
            polygon_intersection_relation = polygon_boundary.intersects(intersection)
            if polygon_intersection_relation.sum() != 2:
                raise ValueError(error_msg + '\nThe corner intersection must intersects with two edges.')
            relation = 'corner'
            edges = polygon_boundary[polygon_intersection_relation].to_list()
            distance = 0.0
        else:
            raise ValueError(error_msg + '\nIntersection is not corner or within edge.')

        return polygon, polygon_boundary, relation, edges, distance

    #切割多边形
    def _slice_polygon(self, polygon: Polygon, intersection: Point, land_use_type: int) -> Polygon:
        """Slice the polygon from the given intersection.

        Args:
            polygon: polygon to be sliced.
            intersection: intersection point.
            land_use_type: land use type of the new land use.

        Returns:
            sliced_polygon: sliced polygon.
        """
        search_max_length = self._required_max_edge_length[land_use_type] + self._common_min_edge_length
        search_max_area = self._required_max_area[land_use_type]
        search_min_area = self._required_min_area[land_use_type]
        polygon, polygon_boundary, relation, edges, distance = self._simplify_polygon(polygon, intersection)
        gdf = self._current_gdf
        all_intersections = gdf[gdf.geom_type == 'Point']
        min_edge_length = self._required_min_edge_length[land_use_type]
        max_edge_length = self._required_max_edge_length[land_use_type]
        if relation == 'edge':
            edge = edges[0]
            land_use_polygon = slice_polygon_from_edge(
                polygon, polygon_boundary, edge, intersection, all_intersections, distance, self.EPSILON,
                self._cell_edge_length, min_edge_length, max_edge_length, search_max_length,
                search_max_area, search_min_area)
        elif relation == 'corner':
            edge_1_intersection = MultiPoint(edges[0].coords).difference(intersection)
            edge_1 = LineString([intersection, edge_1_intersection])
            edge_2_intersection = MultiPoint(edges[1].coords).difference(intersection)
            edge_2 = LineString([intersection, edge_2_intersection])
            land_use_polygon = slice_polygon_from_corner(
                polygon, polygon_boundary, intersection, edge_1, edge_1_intersection, edge_2, edge_2_intersection,
                all_intersections, self.EPSILON, self._cell_edge_length,
                min_edge_length, max_edge_length, search_max_length,
                search_max_area, search_min_area)
        else:
            raise ValueError('Relation must be edge or corner.')

        land_use_polygon = get_intersection_polygon_with_maximum_area(land_use_polygon, polygon)
        return land_use_polygon

    #将处理完后剩余的地块添加回地理数据库中
    def _add_remaining_feasible_blocks(self, feasible_polygon: Polygon, land_use_polygon: Polygon) -> None:
        """Add remaining feasible blocks back to gdf.

        Args:
            feasible_polygon: feasible polygon.
            land_use_polygon: land use polygon.
        """
        intersections = self._gdf[(self._gdf['existence'] == True) & (self._gdf.geom_type == 'Point')].unary_union
        feasible_polygon = snap(feasible_polygon, intersections, self.SNAP_EPSILON/self._cell_edge_length)
        remaining_feasibles = feasible_polygon.difference(land_use_polygon)

        error_msg = 'feasible region: {}'.format(feasible_polygon)
        error_msg += '\nland_use region: {}'.format(land_use_polygon)
        error_msg += '\nremaining feasible region: {}'.format(remaining_feasibles)

        if remaining_feasibles.area > 0:
            if remaining_feasibles.geom_type in ['Polygon', 'MultiPolygon']:
                if remaining_feasibles.geom_type == 'Polygon':
                    remaining_feasibles = MultiPolygon([remaining_feasibles])
                for remaining_feasible in list(remaining_feasibles.geoms):
                    self._update_gdf(
                        remaining_feasible, city_config.FEASIBLE, build_boundary=False, error_msg='Remaining feasible.')
            else:
                raise ValueError(error_msg + '\nRemaining feasible region is neither Polygon nor MultiPolygon.')
        elif not land_use_polygon.equals(feasible_polygon):
            raise ValueError(
                error_msg + '\nThe area of remaining feasible region is 0, but land_use does not equals to feasible.')

    #将简化后的多边形对齐到原图上
    def _simplify_snap_polygon(self, polygon: Polygon) -> Tuple[Polygon, MultiPoint, List]:
        """Simplify the polygon and snap it to existing intersections.

        Args:
            polygon: polygon to be simplified and snapped.

        Returns:
            polygon: the simplified polygon.
            intersections: existing intersections.
            new_intersections: new intersections.
        """
        cached_polygon = polygon
        polygon = polygon.normalize().simplify(self.SNAP_EPSILON/self._cell_edge_length, preserve_topology=True)
        cached_polygon_simplify = polygon
        polygon = simplify_by_distance(polygon, self.EPSILON)
        cached_polygon_simplify_distance = polygon
        existing_intersections = self._gdf[
            (self._gdf.geom_type == 'Point') & (self._gdf['existence'] == True)].unary_union
        polygon = snap(polygon, existing_intersections, self.SNAP_EPSILON/self._cell_edge_length)
        if polygon.is_empty:
            return None, None, None
        if polygon.geom_type != 'Polygon':
            error_msg = 'Original land_use polygon: {}'.format(cached_polygon)
            error_msg += '\nLand_use polygon after simplify: {}'.format(cached_polygon_simplify)
            error_msg += '\nLand_use polygon after simplify by distance: {}'.format(cached_polygon_simplify_distance)
            error_msg += '\nLand_use polygon after snap: {}'.format(polygon)
            raise ValueError(error_msg + '\nLand_use polygon is not a polygon after simplify and snap.')
        intersections = MultiPoint(polygon.exterior.coords[:-1])
        new_intersections = intersections.difference(existing_intersections)
        if new_intersections.is_empty:
            new_intersections = []
        elif new_intersections.geom_type == 'MultiPoint':
            new_intersections = list(new_intersections.geoms)
        elif new_intersections.geom_type == 'Point':
            new_intersections = [new_intersections]
        else:
            error_msg = 'New intersections: {}'.format(new_intersections)
            error_msg += '\nType of new intersections: {}'.format(new_intersections.geom_type)
            raise ValueError(error_msg + '\nThe type of new intersections is not point or multipoint or empty.')
        return polygon, intersections, new_intersections

    #导入简化多边形后，将新的交点添加到地理数据库中
    def _add_new_intersections(self,
                               land_use_polygon: Polygon,
                               intersections: MultiPoint,
                               new_intersections: List) -> None:
        """Add new intersections to gdf.

        Args:
            land_use_polygon: polygon of land use to be updated.
            intersections: existing intersections.
            new_intersections: new intersections.
        """
        if len(new_intersections) == len(intersections.geoms):
            error_msg = 'New intersections:'
            for new_intersection in new_intersections:
                error_msg += '\n{}'.format(new_intersection)
            raise ValueError(error_msg + '\nAll new intersections without any old intersections!')
        for new_intersection in new_intersections:
            intersection_gdf = GeoDataFrame(
                [[self._counter(), city_config.INTERSECTION, True, new_intersection]],
                columns=['id', 'type', 'existence', 'geometry']).set_index('id')
            self._gdf = pd.concat([self._gdf, intersection_gdf])
            roads_or_boundaries = self._gdf[(self._gdf.geom_type == 'LineString') & (self._gdf['existence'] == True)]
            within_existing_roads_or_boundaries = roads_or_boundaries.distance(new_intersection) < self.EPSILON
            if within_existing_roads_or_boundaries.any():
                road_or_boundary_to_split = roads_or_boundaries[within_existing_roads_or_boundaries]
                if len(road_or_boundary_to_split) > 1:
                    error_msg = 'polygon: {}'.format(land_use_polygon)
                    error_msg += '\nnew intersection: {}'.format(new_intersection)
                    error_msg += '\nroad or boundary to split:'
                    for var in range(len(road_or_boundary_to_split)):
                        error_msg += '\n{}'.format(road_or_boundary_to_split['geometry'].iloc[var])
                    error_msg += '\nNew intersection is located at more than 1 existing roads or boundaries.'
                    raise ValueError(error_msg)
                road_or_boundary_to_split_linestring = road_or_boundary_to_split['geometry'].iloc[0]
                road_or_boundary_to_split_type = road_or_boundary_to_split['type'].iloc[0]
                road_or_boundary_1 = LineString([road_or_boundary_to_split_linestring.coords[0], new_intersection])
                road_or_boundary_2 = LineString([road_or_boundary_to_split_linestring.coords[1], new_intersection])
                road_or_boundary_gdf = GeoDataFrame(
                    [[self._counter(), road_or_boundary_to_split_type, True, road_or_boundary_1],
                     [self._counter(), road_or_boundary_to_split_type, True, road_or_boundary_2]],
                    columns=['id', 'type', 'existence', 'geometry']).set_index('id')
                self._gdf = pd.concat([self._gdf, road_or_boundary_gdf])
                road_or_boundary_to_split_id = road_or_boundary_to_split.index[0]
                self._gdf.at[road_or_boundary_to_split_id, 'existence'] = False
            self._gdf['geometry'] = self._gdf['geometry'].apply(lambda x: snap(x, new_intersection, self.EPSILON))

    #导入简化多边形后，将新的边界添加到地理数据库中
    def _add_new_boundaries(self, land_use_polygon: Polygon) -> None:
        """Add new boundaries to gdf.

        Args:
            land_use_polygon: polygon of land use to be updated.
        """
        new_boundaries = get_boundary_edges(land_use_polygon, 'MultiLineString')
        roads_or_boundaries = self._gdf[(self._gdf.geom_type == 'LineString')
                                        & (self._gdf['existence'] == True)].unary_union
        new_boundaries = new_boundaries.difference(roads_or_boundaries)
        if new_boundaries.is_empty:
            new_boundaries = []
        elif new_boundaries.geom_type == 'MultiLineString':
            new_boundaries = list(new_boundaries.geoms)
        elif new_boundaries.geom_type == 'LineString':
            new_boundaries = [new_boundaries]
        else:
            error_msg = 'New boundaries: {}'.format(new_boundaries)
            error_msg += '\nType of new boundaries: {}'.format(new_boundaries.geom_type)
            raise ValueError(error_msg + '\nNew boundaries is not linestring or multilinestring or empty.')

        for new_boundary in new_boundaries:
            if len(new_boundary.coords) > 2:
                error_msg = 'New boundary: {}'.format(new_boundary)
                raise ValueError(error_msg + '\nNumber of coords of new boundary is greater than 2.')
            boundary_gdf = GeoDataFrame(
                [[self._counter(), city_config.BOUNDARY, True, new_boundary]],
                columns=['id', 'type', 'existence', 'geometry']).set_index('id')
            self._gdf = pd.concat([self._gdf, boundary_gdf])

    #导入简化多边形后，将新的土地利用情况添加到地理数据库中
    def _add_land_use_polygon(self, land_use_polygon: Polygon, land_use_type: int) -> None:
        """Add land use polygon to gdf.

        Args:
            land_use_polygon: polygon of land use to be updated.
            land_use_type: land use type of the new land use.
        """
        land_use_gdf = GeoDataFrame(
            [[self._counter(), land_use_type, True, land_use_polygon]],
            columns=['id', 'type', 'existence', 'geometry']).set_index('id')
        land_use_gdf['rect'] = momepy.Rectangularity(land_use_gdf).series
        land_use_gdf['eqi'] = momepy.EquivalentRectangularIndex(land_use_gdf).series
        land_use_gdf['sc'] = momepy.SquareCompactness(land_use_gdf).series
        self._gdf = pd.concat([self._gdf, land_use_gdf])

    #上述步骤完成后，更新地理数据库
    def _update_gdf_without_building_boundaries(self,
                                          land_use_polygon: Polygon,
                                          land_use_type: int,
                                          new_intersections: List,
                                          error_msg: Text = '') -> None:
        """Update the gdf without building boundaries.

        Args:
            land_use_polygon: polygon of land use to be updated.
            land_use_type: land use type of the new land use.
            new_intersections: new intersections.
            error_msg: error message.
        """
        if len(new_intersections) > 0:
            error_msg += '\nUpdate polygon: {}'.format(land_use_polygon)
            raise ValueError(error_msg + '\nUpdate polygon without building boundaries creates new points.')
        self._add_land_use_polygon(land_use_polygon, land_use_type)

    def _update_gdf(self,
                    land_use_polygon: Polygon,
                    land_use_type: int,
                    build_boundary: bool = True,
                    error_msg: Text = '') -> Polygon:
        """Update the GDF.

        Args:
            land_use_polygon: polygon of the new land use.
            land_use_type: land use type of the new land use.
            build_boundary: whether to build boundary.
            error_msg: error message.

        Returns:
            land_use_polygon: polygon of the new land use, might be different from the original one due to snapping.
        """
        land_use_polygon, intersections, new_intersections = self._simplify_snap_polygon(land_use_polygon)
        if land_use_polygon is None:
            error_msg = f'Type {land_use_type}\n'
            raise ValueError(error_msg + 'Empty after simplify and snap.')

        if not build_boundary:
            self._update_gdf_without_building_boundaries(land_use_polygon, land_use_type, new_intersections, error_msg)
            return land_use_polygon

        self._add_new_intersections(land_use_polygon, intersections, new_intersections)
        self._add_new_boundaries(land_use_polygon)
        self._add_land_use_polygon(land_use_polygon, land_use_type)

        return land_use_polygon

   #从图中选择一个可用的区块与交点
    def _get_chosen_feasible_block_and_intersection(self, action: int) -> Tuple[int, int]:
        """Get the chosen feasible block and intersection

        Args:
            action: the chosen graph edge.

        Returns:
            Tuple of the chosen (feasible_block_id, intersection_id).
        """
        #参数：action - 选定的图边缘
        chosen_pair = self._current_graph_edges_with_id[action] #通过参数 action 接收一个整数，这个整数代表图中的一条边
        #函数会从 _current_graph_edges_with_id 映射中查找与这个 action 对应的区块ID和交点ID的配对。
        if self._gdf.loc[chosen_pair[0]]['type'] == city_config.FEASIBLE: #检查这个区块是否被标记为可行（FEASIBLE）。
            ### 函数的末尾进行顺序颠倒的原因是为了确保返回的元组始终是（可行区块ID, 交点ID）的格式。
            ### 目的是为了保持一致性，无论 chosen_pair 中哪个元素是可行区块，都能正确地返回可行区块ID和交点ID。
            ### 这对于后续的逻辑处理和数据结构的一致性非常重要。
            
            #如果 chosen_pair[0]（即第一个元素）代表的区块是可行的，那么函数就返回 (chosen_pair[0], chosen_pair[1])，这符合预期的格式
            return chosen_pair[0], chosen_pair[1]  
        else:
            #如果 chosen_pair[0] 代表的区块不是可行的，那么函数假设 chosen_pair[1]（即第二个元素）是可行区块，
            #并返回 (chosen_pair[1], chosen_pair[0])，以确保可行区块ID始终是返回元组的第一个元素。
            return chosen_pair[1], chosen_pair[0]

    #使用整个可行区块
    def _use_whole_feasible(self, feasible_polygon: Polygon, land_use_type: int) -> Polygon:
        """Use the whole feasible block.

        Args:
            feasible_polygon: polygon of the feasible block.
            land_use_type: land use type of the new land use.
        """
        #参数：feasible_polygon - 可行区块的多边形；land_use_type - 新土地利用的类型。
        land_use_polygon = feasible_polygon 
        land_use_polygon = self._update_gdf(
            land_use_polygon, land_use_type, build_boundary=False, error_msg='Whole feasible.') #调用 _update_gdf 函数来更新GDF
        return land_use_polygon #更新后的土地利用多边形

    #在指定的区块和交点位置放置特定土地利用类型
    #确保了土地利用的放置既符合土地利用类型的面积要求，又考虑到了土地的最佳利用
    def _place_land_use(self, land_use_type: int, feasible_id: int, intersection_id: int) -> Tuple[float, int]:
        """Place the land use at the given action position.

        Args:
          land_use_type: The type of the land use to be placed.
          feasible_id: The id of the feasible block.
          intersection_id: The id of the intersection.

        Returns:
            The area of the land use.
            The actual land use type.
        """
        #参数：land_use_type - 要放置的土地利用类型；feasible_id - 可行区块的ID；intersection_id - 交点的ID。
        actual_land_use_type = land_use_type #定义了一个变量 actual_land_use_type 来存储最终确定的土地利用类型，初始值为输入的 land_use_type。
        feasible_polygon = self._gdf.loc[feasible_id, 'geometry'] #通过 feasible_id 从地理数据框架（GDF）中获取对应的可行区块多边形 feasible_polygon
       
        if feasible_polygon.area*self._cell_area <= self._required_max_area[land_use_type]: ###检查可行区块的面积是否小于或等于该土地利用类型所需的最大面积
            land_use_polygon = self._use_whole_feasible(feasible_polygon, land_use_type) ###如果是，就使用整个区块作为土地利用区域
        else:
            intersection = self._gdf.loc[intersection_id, 'geometry']
            land_use_polygon = self._slice_polygon(feasible_polygon, intersection, land_use_type) #如果不是，选定区块太大了，会尝试将区块切割成更小的部分来满足土地利用的需求
            if land_use_polygon.area < self.EPSILON:  #如果切割后的区块面积太小，无法满足最小面积要求，函数会抛出一个错误
                error_msg = 'feasible polygon: {}'.format(feasible_polygon)
                error_msg += '\nintersection: {}'.format(intersection)
                error_msg += '\nland_use polygon: {}'.format(land_use_polygon)
                raise ValueError(error_msg + '\nThe area of sliced land_use_polygon is near 0.')
            if (feasible_polygon.area - land_use_polygon.area)*self._cell_area <= self._common_min_area: #如果切割后的地块小于或等于公共最小面积，
                land_use_polygon = self._use_whole_feasible(feasible_polygon, land_use_type)  #则函数决定使用整个可行区块。
            else:
                if land_use_polygon.area*self._cell_area < self._required_min_area[land_use_type]: ##如果切割后的地块小于该土地利用类型所需的最小面积，
                    land_use_polygon = self._update_gdf(land_use_polygon, city_config.GREEN_S) ##则将此畸零地块设置为绿地
                    actual_land_use_type = city_config.GREEN_S
                ## 如果切割后的土地利用多边形 land_use_polygon 的面积乘以单元格面积不小于该土地利用类型所需的最小面积
                ## 那么就不需要将土地利用类型更改为 city_config.GREEN_S。在这种情况下，land_use_polygon 
                ## 已经是一个适合的土地利用区域，因此代码直接调用 _update_gdf 函数来更新地理数据框架（GDF）。
                else:
                    land_use_polygon = self._update_gdf(land_use_polygon, land_use_type)

                self._add_remaining_feasible_blocks(feasible_polygon, land_use_polygon)

        self._gdf.at[feasible_id, 'existence'] = False #函数将可行区块在GDF中的存在标记为 False，表示该区块已被使用。

        land_use_area = land_use_polygon.area*self._cell_area
        return land_use_area, actual_land_use_type  #土地利用的面积和实际土地利用类型的元组

    #在给定的动作位置放置土地利用
    def place_land_use(self, land_use: Dict, action: int) -> None:
        """Place the land use at the given action position.

        Args:
          land_use: A dict containing the type, x, y, area, width and height of the current land use.
          action: The action to take (an integer indicating the chosen graph edge).

        Returns:
            True if the land_use is successfully placed, False otherwise.
        """
        #参数：land_use - 包含当前土地利用类型、x、y、面积、宽度和高度的字典；action - 要采取的动作（指示选定图边缘的整数）。
        feasible_id, intersection_id = self._get_chosen_feasible_block_and_intersection(action)  #使用 _get_chosen_feasible_block_and_intersection 函数来获取可行区块和交点
        land_use_area, actual_land_use_type = self._place_land_use(land_use['type'], feasible_id, intersection_id) #调用 _place_land_use 函数来放置土地利用类型，并更新统计数据。
        self._update_stats(actual_land_use_type, land_use_area)

    #选择一个边界
    def _get_chosen_boundary(self, action: int) -> int:
        """Get the chosen boundary.

        Args:
            action: the chosen graph node. ##参数 action 是一个整数，代表图中的一个节点

        Returns:
            The chosen boundary.
        """
        chosen_boundary = self._current_graph_nodes_id[action]
        if self._gdf.loc[chosen_boundary, 'type'] != city_config.BOUNDARY:
            raise ValueError('The build road action is not boundary node.')
        return chosen_boundary

    # 用于在地理数据框架（GDF）中建立道路
    def build_road(self, action: int) -> None:
        """Build the road at the given action position.

        Args:
          action: The action to take (the chosen node to build road).

        Returns:
            True if the road is successfully built, False otherwise.
        """
        chosen_boundary = self._get_chosen_boundary(action) # 调用 _get_chosen_boundary 方法来获取选定的边界
        self._gdf.loc[chosen_boundary, 'type'] = city_config.ROAD # 将该边界的类型更改为道路

    def get_requirements(self) -> Tuple[np.ndarray, np.ndarray]:  #获取道路规划的要求，包括土地使用比例和数量
        """Get the planning requirements.

        Returns:
            A tuple of the requirements of land_use ratio and land_use count.
        """
        return self._required_plan_ratio, self._required_plan_count

    def get_plan_ratio_and_count(self) -> Tuple[np.ndarray, np.ndarray]:  #获取当前场地的道路规划的信息，包括当前道路规划的土地使用比例和数量
        """Get the planning ratio and count.

        Returns:
            A tuple of the ratio and count of land_use.
        """
        return self._plan_ratio, self._plan_count

    @staticmethod
    def _get_road_boundary_graph(gdf) -> nx.MultiGraph:
        """Return the road and boundary graph."""
        # 筛选出类型为道路（city_config.ROAD）或边界（city_config.BOUNDARY）的 GeoDataFrame 记录
        road_boundary_gdf = gdf[(gdf['type'] == city_config.ROAD) | (gdf['type'] == city_config.BOUNDARY)]
        # 使用 momepy.gdf_to_nx 函数将这些记录转换为 NetworkX 图形。这个图形可以用于进一步的分析和规划
        road_boundary_graph = momepy.gdf_to_nx(road_boundary_gdf.reset_index(), approach='primal', length='length')
        return road_boundary_graph

    @staticmethod
    def _get_domain_features(gdf: GeoDataFrame) -> np.ndarray: #从地理数据集GeoDataFrame中提取特征
        """Get the domain knowledge features.

        Args:
            gdf: the GeoDataFrame.  # 接受一个 GeoDataFrame 作为参数，并返回一个 NumPy 数组

        Returns:
            The domain knowledge features.
        """

        #提取了 ‘rect’（矩形）、‘eqi’（环境质量指数）、‘sc’（社会连通性）这三个特征，并将缺失值填充为 0.5。
        domain_gdf = gdf[['rect', 'eqi', 'sc']].fillna(0.5) #暂时赋值为0.5
        domain_features = domain_gdf.to_numpy()
        return domain_features

    def get_graph_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,  ##从地理数据集GeoDataFrame中提取特征
                                          np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the graph features.  #返回一个包含多个图形特征的元组

        Returns:
            A tuple of the graph features which contains the followings.
            1. node type: the type of the nodes.节点类型（node type）
            2. node coordinates: the x-y coordinate of the nodes.节点坐标（node coordinates）
            3. node area: the area of the nodes.节点面积（node area）
            4. node length: the length of the nodes. 节点长度（node length）
            5. node width: the width of the nodes.节点宽度（node width）
            6. node height: the height of the nodes.节点高度（node height）
            7. edges: the adjacency list.边缘列表（edges）
        """
         #这些特征是通过处理 self._current_gdf（当前的 GeoDataFrame）和 self._current_graph（当前的图形结构）来获得的。                                   
        gdf = self._current_gdf
        graph = self._current_graph
        #节点类型（node type）：使用 gdf['type'].to_numpy(dtype=np.int32) 将节点类型转换为整数类型的 NumPy 数组。                                      
        node_type = gdf['type'].to_numpy(dtype=np.int32) 
        #节点坐标（node coordinates）：计算每个节点的 x 和 y 坐标，并将其标准化到网格大小，使用 np.column_stack 将它们组合成一个数组。                                      
        node_coordinates = np.column_stack((gdf.centroid.x/self._grid_cols, gdf.centroid.y/self._grid_rows))
        #节点面积（node area）：计算每个节点的面积，并将其乘以单元格面积 self._cell_area 转换为实际面积。                                      
        node_area = gdf.area.to_numpy(dtype=np.float32)*self._cell_area
        #节点长度（node length）：计算每个节点的长度，并将其乘以单元格边长 self._cell_edge_length 转换为实际长度。                                      
        node_length = gdf.length.to_numpy(dtype=np.float32)*self._cell_edge_length
        #节点宽度（node width）和 节点高度（node height）：使用 gdf.bounds 计算每个节点的宽度和高度，并进行相应的单位转换。                                      
        bounds = gdf.bounds
        node_width = (bounds['maxx'] - bounds['minx']).to_numpy(dtype=np.float32)*self._cell_edge_length
        node_height = (bounds['maxy'] - bounds['miny']).to_numpy(dtype=np.float32)*self._cell_edge_length
        #节点领域特征（node domain）：调用 _get_domain_features 方法获取节点的领域特征。                                      
        node_domain = self._get_domain_features(gdf)
        #边缘列表（edges）：使用 np.array(graph.edges) 获取图形中所有边缘的列表。
        edges = np.array(graph.edges)

        return node_type, node_coordinates, node_area, node_length, node_width, node_height, node_domain, edges

    def _get_road_graph(self) -> nx.MultiGraph: #从地理数据库中获取道路信息，将其转换为graph图
        """Return the road graph."""
        road_gdf = self._gdf[(self._gdf['type'] == city_config.ROAD) & (self._gdf['existence'] == True)]
        road_graph = momepy.gdf_to_nx(road_gdf, approach='primal', length='length', multigraph=False)
        return road_graph

    def get_road_network_reward(self) -> Tuple[float, Dict]: #计算道路网络的奖励值
        """Get the road network reward.

        Returns:
            The road network reward.
        """
        gdf = self._gdf[self._gdf['existence'] == True]
        road_graph = self._get_road_graph()

        # connectivity of road network 基于道路连通性进行奖励，连通性越高奖励越高
        connectivity_reward = 1.0/nx.number_connected_components(road_graph)

        # density of road network 基于路网密度进行奖励，密度越高奖励越高
        road_length = gdf[gdf['type'] == city_config.ROAD].length
        road_total_length_km = road_length.sum()*self._cell_edge_length/1000
        community_area_km = self._community_area/1000/1000
        road_network_density = road_total_length_km/community_area_km
        density_reward = road_network_density/10.0

        # dead end penalty 发现一个断头路就惩罚一分
        degree_sequence = np.array([d for n, d in road_graph.degree()], dtype=np.int32)
        num_dead_end = np.count_nonzero(degree_sequence == 1)
        dead_end_penalty = 1.0/(num_dead_end + 1)

        # penalty for short/long road 有一条长度小于100的道路就惩罚一分，有一条长度大于600的道路就惩罚一分
        road_gdf = gdf[gdf['type'] == city_config.ROAD]
        road_gdf = momepy.remove_false_nodes(road_gdf)
        road_length = road_gdf.length
        num_short_roads = len(road_length[road_length*self._cell_edge_length < 100])
        short_road_penalty = 1.0/(num_short_roads + 1)
        num_long_roads = len(road_length[road_length*self._cell_edge_length > 600])
        long_road_penalty = 1.0/(num_long_roads + 1)

        # penalty for road distance 发现一个大街区（有一边长度超过800），就惩罚一分。
        road_gdf = gdf[gdf['type'] == city_config.ROAD]
        blocks = polygonize(road_gdf['geometry'])
        block_bounds = [block.bounds for block in blocks]
        block_width_height = np.array([(b[2] - b[0], b[3] - b[1]) for b in block_bounds], dtype=np.float32)
        num_large_blocks = np.count_nonzero(
            np.logical_or(
                block_width_height[:, 0]*self._cell_edge_length > 800,
                block_width_height[:, 1]*self._cell_edge_length > 800))
        road_distance_penalty = 1.0/(num_large_blocks + 1)

        road_network_reward = 1.0 * connectivity_reward + 1.0 * density_reward + 1.0 * dead_end_penalty + \
            1.0 * short_road_penalty + 1.0 * long_road_penalty + 1.0 * road_distance_penalty # 计算道路网络奖励的各个组成部分
        road_network_reward = road_network_reward/6.0 # 将总奖励值平均化
        info = {'connectivity_reward': connectivity_reward, # 创建包含所有评价指标的字典
                'density_reward': density_reward,
                'dead_end_penalty': dead_end_penalty,
                'short_road_penalty': short_road_penalty,
                'long_road_penalty': long_road_penalty,
                'road_distance_penalty': road_distance_penalty}

        return road_network_reward, info

    def get_life_circle_reward(self, weight_by_area: bool = False) -> Tuple[float, Dict]:  
        #评估居民区与公共服务设施之间的距离和分布情况得到生活圈奖励值

        """Get the reward of the life circle.

        Returns:
            The reward of the life circle.
        """
        gdf = self._gdf[self._gdf['existence'] == True] #从数据库中选出存在土地利用的区块
        residential_centroid = gdf[gdf['type'] == city_config.RESIDENTIAL].centroid #计算居住区的中心点与居住区面积
        residential_area = gdf[gdf['type'] == city_config.RESIDENTIAL].area.to_numpy() #初始化最小公服距离与公服之间的距离
        num_public_service = 0 #初始化公服数量
        minimum_public_service_distances = [] # 最小公共服务距离
        public_service_pairwise_distances = [] # 公共服务之间的距离
        public_service_area = 0.0 #初始化公服的面积
        for public_service in city_config.PUBLIC_SERVICES_ID: # 遍历 city_config.PUBLIC_SERVICES_ID 中定义的所有公共服务类型
            if not isinstance(public_service, tuple):
                public_service_gdf = gdf[gdf['type'] == public_service] #如果不是元组（说明只存在一种数据，即存在一种公服设施），会直接选择该公服
            else:
                public_service_gdf = gdf[gdf['type'].isin(public_service)] #如果是元组（说明包括多种数据，即存在多种类型的公服），则会选择所有类型的公服
            public_service_centroid = public_service_gdf.centroid.unary_union #同时计算每种公共服务的中心点，并将该区块内所有公服的中心点合并为一个点

            num_same_public_service = len(public_service_gdf) #计算同一类型的公共服务设施的数量
            if num_same_public_service > 0:  # 如果存在至少一个此类公共服务设施
                distance = residential_centroid.distance(public_service_centroid).to_numpy() #计算居民区中心点到公共服务设施中心点的距离
                minimum_public_service_distances.append(distance) # 将计算出的距离添加到 minimum_public_service_distances 列表中。
                num_public_service += 1 # 公共服务数量加一
                public_service_area += public_service_gdf.area.sum()*self._cell_area # 计算所有公共服务设施的总面积并乘以单元格面积，累加到 public_service_area。

                if num_same_public_service > 1: # 如果存在多于一个同类型的公共服务设施
                    # 提取所有公共服务设施中心点的x和y坐标
                    public_service_x = public_service_gdf.centroid.x.to_numpy()
                    public_service_y = public_service_gdf.centroid.y.to_numpy()
                    # 将x和y坐标数组合并为一个二维数组，其中每个公共服务设施的坐标是数组中的一个元素
                    public_service_xy = np.stack([public_service_x, public_service_y], axis=1)
                    # 计算所有公共服务设施中心点之间的距离
                    pair_distance = cdist(public_service_xy, public_service_xy)
                    # 计算所有距离的平均值，该值表示多个同类型的公服到居住区的距离
                    average_pair_distance = np.mean(pair_distance[pair_distance > 0])
                    public_service_pairwise_distances.append(average_pair_distance)

        if num_public_service > 0: #如果公服数量大于0
            public_service_distance = np.column_stack(minimum_public_service_distances) # 将每个居住区中心点到最近公服的距离合成一个数组
            # 通过乘以单元格边长并与特定距离阈值（例如1000米、500米、300米）比较，计算不同时间范围内可达的公共服务设施的比例。
            life_circle_15min = np.count_nonzero(
                public_service_distance*self._cell_edge_length <= 1000, axis=1)/num_public_service
            life_circle_10min = np.count_nonzero(
                public_service_distance*self._cell_edge_length <= 500, axis=1)/num_public_service
            life_circle_5min = np.count_nonzero(
                public_service_distance*self._cell_edge_length <= 300, axis=1)/num_public_service
            # 如果 weight_by_area 为 False，则计算10分钟生活圈的平均值作为效率奖励
            # weight_by_area为False意味着在计算效率奖励时，不会考虑每个居民区域的面积大小。
            # 换句话说，所有居民区域在计算10分钟生活圈的平均值时被视为同等重要，不会根据它们的面积大小给予不同的权重。
            # 如果weight_by_area为True，则意味着在计算效率奖励时，会根据居民区域的面积大小进行加权，
            # 这样可以更准确地反映出大面积居民区域对于生活圈可达性的贡献。
            if not weight_by_area:
                efficiency_reward = life_circle_10min.mean()
            else:
                efficiency_reward = np.average(life_circle_10min, weights=residential_area)
            # 去中心化奖励：计算公服之间的平均距离与参考距离（reference_distance）的比值作为去中心化奖励，确保公服是均匀分布而不是集中分布于某点
            # reference_distance 是一个参考距离，它是通过计算网格的列数（self._grid_cols）和行数（self._grid_rows）的平方和的平方根得到的。
            # 这个距离可以看作是网格的对角线长度，代表了网格的最大可能距离。
            reference_distance = math.sqrt(self._grid_cols**2 + self._grid_rows**2)
            decentralization_reward = np.array(public_service_pairwise_distances).mean()/reference_distance
            # 公服面积占比奖励：计算公共服务设施面积与社区面积的比例作为奖励
            utility_reward = public_service_area/self._community_area
            # 总奖励：将效率奖励与去中心化奖励的5%相加得到总奖励
            reward = efficiency_reward + 0.05 * decentralization_reward
            # 创建一个包含各种奖励和可达性指标的字典
            info = {'life_circle_15min': life_circle_15min.mean(),# 表示居民在15分钟内可达的公共服务设施的平均比例
                    'life_circle_10min': life_circle_10min.mean(),# 表示居民在10分钟内可达的公共服务设施的平均比例
                    'life_circle_5min': life_circle_5min.mean(),# 表示居民在5分钟内可达的公共服务设施的平均比例
                     # 'life_circle_10min_area': 表示按居民区域面积加权后，居民在10分钟内可达的公共服务设施的平均比例
                    'life_circle_10min_area': np.average(life_circle_10min, weights=residential_area),
                    'decentralization_reward': decentralization_reward,
                    'utility': utility_reward}
            # 计算所有居民在10分钟内可达的公共服务设施的比例，并将这些比例添加到信息字典中
            # 可以根据研究侧重点选择5分钟或者15分钟
            life_circle_10min_all = np.count_nonzero(
                public_service_distance*self._cell_edge_length <= 500, axis=0)/public_service_distance.shape[0]
            for index, service_name in enumerate(city_config.PUBLIC_SERVICES):
                info[service_name] = life_circle_10min_all[index]
            return reward, info
        else:
            return 0.0, dict()

    def get_greenness_reward(self) -> float: # 计算城市规划中绿化覆盖对居民区的奖励值
        """Get the reward of the greenness.

        Returns:
            The reward of the greenness.
        """
        gdf = self._gdf[self._gdf['existence'] == True] # 从数据集中选择数据
        green_id = city_config.GREEN_ID #从城市配置（city_config）中获取绿色空间类型的标识符列表，并将其赋值给变量green_id。
        
        # gdf['type'].isin(green_id)：这部分检查gdf中的每个元素的类型是否包含在green_id列表中。
        # green_id是预先定义的绿色空间类型的标识符列表，
        # 例如公园、草地等。#gdf.area*self._cell_area >= city_config.GREEN_AREA_THRESHOLD：这部分
        # 计算每个元素的面积（gdf.area）乘以单元格面积，（self._cell_area）并检查结果是否大于或等于一个
        # 预设的阈值（city_config.GREEN_AREA_THRESHOLD）。这个阈值是用来确定一个元素是否足够大，
        # 以被认为是有效的绿色空间
        green_gdf = gdf[(gdf['type'].isin(green_id)) & (gdf.area*self._cell_area >= city_config.GREEN_AREA_THRESHOLD)]
        green_cover = green_gdf.buffer(300/self._cell_edge_length).unary_union  # 计算绿地300米缓冲区
        residential = gdf[gdf['type'] == city_config.RESIDENTIAL].unary_union #找到居民区类型的地理元素，并将其合并
        green_covered_residential = green_cover.intersection(residential) # 计算绿色覆盖区域与居民区域的交集，得到被绿色空间覆盖的居民区域
        reward = green_covered_residential.area / residential.area # 计算被绿色空间覆盖的居民区域面积与居民区总面积的比例，作为绿化奖励
        return reward

    def get_concept_reward(self) -> Tuple[float, Dict]: #规划概念奖励
        """Get the reward of the planning concept.

        Returns:
            The reward of the concept.
            The information of the concept reward.
        """
        if len(self._concept) == 0: # 首先检查概念列表 _concept 是否为空。如果为空，则抛出一个值错误
            raise ValueError('The concept list is empty.')
        #从地理数据框 gdf 中选择存在且几何类型为多边形的元素
        gdf = self._gdf[(self._gdf['existence'] == True) & (self._gdf.geom_type == 'Polygon')]
        reward = 0.0 # 初始化奖励 reward 为0.0
        info = dict() # 创建一个空字典 info
        # 遍历概念列表 _concept，根据概念的类型（'center' 或 'axis'）计算奖励和信息
        for i, concept in enumerate(self._concept):
            # 如果概念类型是 'center'，调用 _get_center_concept_reward_info 函数计算中心概念的奖励和信息，
            # 然后将结果累加到 reward 变量，并将信息添加到 info 字典
            if concept['type'] == 'center':
                center_reward, center_info = self._get_center_concept_reward_info(gdf, concept)
                reward += center_reward
                info['{}_center'.format(i)] = center_info
            # 如果概念类型是 'axis'，调用 _get_axis_concept_reward_info 函数计算轴线概念的奖励和信息，
            # 同样累加和添加到 reward 和 info
            elif concept['type'] == 'axis':
                axis_reward, axis_info = self._get_axis_concept_reward_info(gdf, concept)
                reward += axis_reward
                info['{}_axis'.format(i)] = axis_info
            else:
                raise ValueError(f'The concept type {concept["type"]} is not supported.')
        # 将累加的奖励 reward 除以概念列表的长度，得到平均奖励，并返回这个奖励和信息字典 info
        reward /= len(self._concept)
        return reward, info

   # 计算规划方案中中心概念的奖励值
    def _get_center_concept_reward_info(self, gdf: GeoDataFrame, concept: Dict) -> Tuple[float, Dict]:
        """Get the reward of the center concept.

        Args:
            gdf: The GeoDataFrame of the city.
            concept: The concept.

        Returns:
            The reward of the center concept.
            The information of the center concept.
        """
        # 从 concept 字典中获取中心点的几何信息。这个中心点代表了城市规划中心概念的地理位置
        center = concept['geometry']
        # 距离阈值 (distance_threshold): 这是一个数值，表示从中心点向外延伸的距离，在这个距离内的土地使用被认为与城市中心概念相关
        distance_threshold = concept['distance']
        # 中心圆形区域 (center_circle): 使用中心点的几何信息和距离阈值创建一个圆形区域
        center_circle = center.buffer(distance_threshold/self._cell_edge_length)
        # 从地理数据框 gdf 中筛选出与中心圆形区域相交的元素。这些元素代表了在城市中心概念范围内的土地使用
        center_gdf = gdf[gdf.intersects(center_circle)]
        # 这是一个列表，包含了与城市中心概念相关的土地使用类型，如商业、住宅等
        related_land_use = concept['land_use']
        # 从 center_gdf 中筛选出类型在 related_land_use 列表中的元素。这些元素代表了与城市中心概念直接相关的土地使用
        center_related_gdf = center_gdf[center_gdf['type'].isin(related_land_use)]
        # 计算与城市中心概念相关的土地使用元素的数量与中心圆形区域内所有土地使用元素数量的比例。
        # 这个比例作为奖励值 (reward)，反映了城市中心概念的实现程度。
        center_related_land_use_ratio = len(center_related_gdf)/len(center_gdf)
        reward = center_related_land_use_ratio

        # 创建一个包含中心概念相关信息的字典，如中心点坐标、距离阈值、相关土地使用类型和比例
        info = dict()
        info['center'] = (center.x, center.y)
        info['distance_threshold'] = distance_threshold
        info['related_land_use'] = related_land_use
        info['related_land_use_ratio'] = center_related_land_use_ratio
        return reward, info

    
    # 为城市规划中的轴线概念计算奖励值
    def _get_axis_concept_reward_info(self, gdf: GeoDataFrame, concept: Dict) -> Tuple[float, Dict]:
        """Get the reward of the axis concept.

        Args:
            gdf: The GeoDataFrame of the city.
            concept: The concept.

        Returns:
            The reward of the axis concept.
            The information of the axis concept.
        """
        axis = concept['geometry'] # 从 concept 字典中提取轴线的几何信息
        distance_threshold = concept['distance'] # 从 concept 字典中获取轴线概念的距离阈值
        # 使用 axis.buffer 方法创建一个围绕轴线的带状区域，其宽度由距离阈值决定
        axis_band = axis.buffer(distance_threshold/self._cell_edge_length, cap_style=2, join_style=2)
        # 从地理数据框 gdf 中筛选出与轴线带区域相交的元素
        axis_gdf = gdf[gdf.intersects(axis_band)]
        # 从 concept 字典中获取与轴线概念相关的土地使用类型，如商业、住宅等。
        related_land_use = concept['land_use']
        # 进一步筛选出类型符合 related_land_use 的元素，代表了与城市中心概念直接相关的土地使用
        axis_related_gdf = axis_gdf[axis_gdf['type'].isin(related_land_use)]
        if len(axis_related_gdf) > 0: # 如果有相关土地使用元素存在，计算四个数据
            
            # 相关土地使用元素与轴线带区域内所有元素的数量比
            axis_related_land_use_ratio = len(axis_related_gdf)/len(axis_gdf)
            
            # 不同类型的相关土地使用元素的数量与预期类型数量的比
            axis_related_land_use_type = axis_related_gdf['type'].nunique()/len(related_land_use)

            # 计算与城市规划轴线概念相关的土地使用元素在轴线上的分布范围
            # 这个变量存储了每个相关土地使用元素的质心在轴线上的投影位置。这里使用了 normalized=True 参数，
            # 意味着投影位置是以轴线长度为单位的相对值，范围在0到1之间。
            axis_related_land_use_project = axis_related_gdf.centroid.apply(lambda x: axis.project(x, normalized=True))
            # 计算了所有投影位置的最大值和最小值之差，得到的结果表示相关土地使用元素在轴线上的投影范围。
            # 这个范围可以用来评估相关土地使用元素在轴线上的分布是否集中或分散
            axis_related_land_use_expand = axis_related_land_use_project.max() - axis_related_land_use_project.min()
            # 奖励值 (reward) 是上述三个比例的平均值
            reward = (axis_related_land_use_ratio + axis_related_land_use_type + axis_related_land_use_expand)/3
            
            # 创建信息字典 (info): 包含轴线坐标、距离阈值、相关土地使用类型及上述计算的比例
            info = dict()
            info['axis'] = axis.coords[:]
            info['distance_threshold'] = distance_threshold
            info['related_land_use'] = related_land_use
            info['related_land_use_ratio'] = axis_related_land_use_ratio
            info['related_land_use_type'] = axis_related_land_use_type
            info['related_land_use_expand'] = axis_related_land_use_expand
        else:
            reward = 0.0
            info = dict()
            info['axis'] = axis.coords[:]
            info['distance_threshold'] = distance_threshold
            info['related_land_use'] = related_land_use
            info['related_land_use_ratio'] = 0.0
            info['related_land_use_type'] = 0.0
            info['related_land_use_expand'] = 0.0

        return reward, info
