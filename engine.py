import numpy as np
from numbers import Number
from typing import Union, Any
import math

class Point():
    def __init__(self, x, y) -> None:
        self.pos = np.array([float(x), float(y)])
    
    def __call__(self) -> np.ndarray:
        return self.pos
    
    def __getitem__(self, index):
        return self.pos[index]
    
    def distance(self, p: 'Point'):
        return np.abs(np.hypot(p.x - self.x, p.y - self.y))
    
    @property
    def x(self):
        return self.pos[0]
    @property
    def y(self):
        return self.pos[1]
    
    @x.setter
    def x(self,x):
        self.pos[0] = x
    @y.setter
    def y(self, y):
        self.pos[1] = y

    def __repr__(self) :
        return f"Point({self.x}, {self.y})"
    
    def __add__(self, other) -> 'Point':
        return Point(self.x + other[0], self.y + other[1])
    def __radd__(self, other) -> 'Point':
        return self + [other, other]
    
    def __sub__(self, other) -> 'Point':
        return Point(self.x - other[0], self.y - other[1])
    def __rsub__(self, other) -> 'Point':
        other = float(other)
        return self - [other, other]
    
    def __mul__(self, other) -> 'Point':
        return Point(self.x * other[0], self.y * other[1])
    def __rmul__(self, other) -> 'Point':
        other = float(other)
        return self * [other, other]
    
    def __truediv__(self, other) -> 'Point':
        return Point(self.x / other, self.y / other)
    def __rdiv__(self, other) -> 'Point':
        return self / [other, other]

class Vector():
    def __init__(self, x: Number, y: Number) -> None:

        self.value = np.array([x, y])

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    @property
    def x(self):
        return self.value[0]
    
    @property
    def y(self):
        return self.value[1]
    @x.setter
    def x(self,x):
        self.value[0] = x
    
    @y.setter
    def y(self, y):
        self.value[1] = y

    @property
    def length(self) -> Number:
        return np.abs(math.hypot(self.x, self.y))
    
    @property
    def unit_vector(self) -> np.ndarray:
        return self.value / self.length
    
   
class Line():
    def __init__(self, p1: Point, p2: Point) -> None:
        self.type = 'line'
        if type(p1) != Point:
            p1 = Point(p1[0], p1[1])
        if type(p2) != Point:
            p2 = Point(p2[0], p2[1])
        self.p1, self.p2 = p1, p2
    def __repr__(self):
        return f"Line({self.p1}, {self.p2})"
    
    @property
    def x(self) -> Number:
        return self.p2.x - self.p1.x
        
    @property
    def y(self) -> Number:
        return self.p2.y - self.p1.y
    
    @property
    def vec(self) -> Vector:
        return Vector(self.x, self.y)
        
    @property
    def length(self) -> Number:
        return self.p1.distance(self.p2)
    
    @property
    def unit_vector(self) -> np.ndarray:
        return self.vec.value / self.length
    
    def closest_point(self, p: Point):
        (x1, y1), (x2, y2), (x3, y3) = self.p1.pos, self.p2.pos, p.pos
        dx, dy = x2-x1, y2-y1
        det = dx*dx + dy*dy 
        a = (dy*(y3-y1)+dx*(x3-x1))/det
        return Point(x1+a*dx, y1+a*dy)
    
    def intersection_point(self, line: 'Line'):
        (x1,y1), (x2,y2), (x3,y3), (x4,y4) = self.p1, self.p2, line.p1, line.p2
        det = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if det != 0:
            px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / det
            py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / det
        
            return Point(px, py)
        else:
            return False

    def intersection(self, line: 'Line'):
        x1, y1 = self.p1
        x2, y2 = self.p2
        x3, y3 = line.p1
        x4, y4 = line.p2

        # Calculate the slopes and y-intercepts of the two lines
        m1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - m1 * x1
        m2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - m2 * x3

        # Check if the lines are parallel
        if m1 == m2:
            return None

        # Calculate the intersection point
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

        return x, y

    
    def point_on_line(self, point: Point) -> bool:
        cumulative_dist_to_point = point.distance(self.p1) + point.distance(self.p2)

        return math.isclose(cumulative_dist_to_point, self.length)
    
    def move_from_point(self, point: Point, length: Number) -> Point:
        return point - self.unit_vector * length        



class Circle():
    def __init__(self, x: Number=0, y: Number=0, radius: Number=1, mass: Number=1, vx:Number=0, vy:Number=0, rot:Number=0) -> None:
        self.type = 'circle'
        self.pos = Point(x, y)
        self.radius = radius
        self.mass = mass
        self.vel = np.array([vx, vy])
        self.forces = []
        self.rot = 0
    
    @property
    def x(self):
        return self.pos.x
    @property
    def y(self):
        return self.pos.y
    @property
    def vx(self):
        return self.vel[0]
    @property
    def vy(self):
        return self.vel[1]
    
    @x.setter
    def x(self,x):
        self.pos[0] = x
    @y.setter
    def y(self, y):
        self.pos[1] = y
    @vx.setter
    def vx(self,vx):
        self.vel[0] = vx
    @vy.setter
    def vy(self, vy):
        self.vel[1] = vy

    def apply_force(self, force: list[list[Number]],timestep=1):
        self.vel = self.vel + np.array(force) / self.mass * timestep
    
    def apply_forces(self, timestep=1):
        for force in self.forces:
            self.apply_force(force, timestep)
    
    def apply_velocity(self, timestep):
        self.pos = self.next_pos(timestep)
    
    def next_pos(self, timestep: Number) -> np.ndarray:
        return self.pos + (self.vel * timestep) / self.mass
    
    def movement_dir(self, timestep) -> Line:
        return Line(self.pos, self.next_pos(timestep))

    def line_collision(self, line: Line, timestep: Number):
        # check if collision is possible
        # https://ericleong.me/research/circle-line/
        
        movement_dir = self.movement_dir(timestep)

        if movement_dir.length == 0:
            return False, False, False
        
        # cases where a collision could happen:
        could_collide = False
        # The point of intersection between line and the movement vector of the circle. (a)
        p_a = line.intersection_point(movement_dir)
        if p_a == False:

            return False, False, False

        if (line.point_on_line(p_a) and movement_dir.point_on_line(p_a)):
            could_collide = True

        # The closest point on the line to the endpoint of the movement vector of the circle. (b)
        p_b = line.closest_point(movement_dir.p2)
        if p_b.distance(movement_dir.p2) < self.radius:
            could_collide = True


        # The closest point on the movement vector to (x1, y1). (c)
        p_c = movement_dir.closest_point(line.p1)
        if p_c.distance(line.p1) < self.radius and movement_dir.point_on_line(p_c):
            could_collide = True


        # The closest point on the movement vector to the other endpoint. (d)
        p_d = movement_dir.closest_point(line.p2)
        if p_c.distance(line.p2) < self.radius and movement_dir.point_on_line(p_d):
            could_collide = True

        if could_collide == False:
            return False, False, False

        # the circle collided
        p_1 = line.closest_point(self.pos)
        collision_point = p_a - self.radius * (p_a.distance(self.pos) / p_1.distance(self.pos)) * movement_dir.unit_vector

        p_c = line.closest_point(p_a)
        if line.point_on_line(p_c):
            # line intersection        
            p_3 = collision_point + (p_1 - p_c)
            direction = self.pos - 2 * (p_3 - collision_point) - collision_point
            direction_norm = Vector(direction[0], direction[1]).unit_vector
            new_vel = direction_norm * movement_dir.length / timestep
            
            print(self.vel, new_vel)
        else:
            # edge intersection
            
            # calculate what endpoint the circle collided with 
            if p_c.distance(line.p1) < p_c.distance(line.p2):
                endpoint = line.p1
            else:
                endpoint = line.p2
            
            closest_point = movement_dir.closest_point(endpoint)
            distance = closest_point.distance(endpoint)  
            
            move_back_distance = math.sqrt(self.radius**2 - distance**2)
            collision_point: Point = movement_dir.move_from_point(collision_point, -move_back_distance)
            
            
            movement_dir_norm = Line(endpoint, collision_point).unit_vector
            new_vel = movement_dir_norm * movement_dir.vec.value

        return True, new_vel, collision_point
    
                

# can have multiple circles
class PhysicsObject():
    def __init__(self, circles: np.ndarray) -> None:
        self.circles = circles
            
        
class PhysicsEnvironment():
    def __init__(self, sizex, sizey, objects=[], lines=[]) -> None:
        self.size : list = [sizex, sizey]
        self.objects :list[PhysicsObject, Circle, Line] = objects
        self.lines : list[Line] = lines
        self.lines += [Line([0,0], [sizex, 0]), Line([sizex, 0], [sizex, sizey]), Line([sizex, sizey], [0, sizey]), Line([0, sizey], [0, 0]), ]

    def run_tick(self, timestep=1):
        
        collision_solver_dict = {}
        for i in range(len(self.objects)):
            collision_solver_dict[self.objects[i]] = self.objects[i+1::]
            
        # filter all the times that can not be in range of the obj
        collisions_dict : dict[Circle: list[Circle]]  = {}
        for circle in list(collision_solver_dict):
            circle: Circle
            collisions_dict[circle] = []
            for circle2 in collision_solver_dict[circle]:
                if circle != circle2:
                    collided, norm, depth = self.circle_circle_collision(circle, circle2)
                    
                    if (collided):
                        collisions_dict[circle].append({'obj': circle2, 'norm': norm, 'depth': depth})         

                        # move them out of eachother
                        circle.move(-norm * depth * 0.51)
                        circle2.move(norm * depth * 0.5)
                        
                        u1, u2 = self.circle_collision(circle, circle2)

                        # add forces
                        circle.forces.append(u1)
                        circle2.forces.append(u2)
                        
                        
            for line in self.lines:
                collided, new_vel, pos = circle.line_collision(line, timestep)
                if collided:
                    circle.vel = new_vel
                    # circle.pos = pos

            
            
        for obj in self.objects:
            obj.forces.append([0, -1])
            obj.apply_forces(timestep)
            obj.apply_velocity(timestep)
            obj.forces = []
    
    


        