from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from math import *
import numpy as np

class Algorithms:
    def __init__(self):
        pass
    
    def calculateDistance (self, p1: QPointF, p2: QPointF):
        dist = sqrt((p1.x()-p2.x())**2+(p1.y()-p2.y())**2)
        
        return dist
    
    def get_point_location(self, q:QPointF, p1:QPointF, p2:QPointF):
        # Calculate determinant for half-plane test
        det = (p2.x() - p1.x())*(q.y() - p1.y()) - (p2.y() - p1.y())*(q.x() - p1.x())
        return 1 if det > 0 else -1 if det < 0 else 0
    
    def get2VectorsAngle(self, p1:QPointF, p2:QPointF, p3:QPointF, p4:QPointF):
        """
        Compute angle between two vectors
        """
        ux = p2.x() - p1.x()
        uy = p2.y() - p1.y()
        
        vx = p4.x() - p3.x()
        vy = p4.y() - p3.y()
        
        #Dot product
        uv = ux * vx + uy * vy
        
        # Norms u, v
        norm_u = sqrt(ux**2 + uy**2)
        norm_v = sqrt(vx**2 + vy**2)
        
        # Prevent division by zero
        if norm_u == 0 or norm_v == 0:
            return 0
        arg = uv/(norm_u * norm_v)
        
        # Correct argument to interval <-1, 1>
        arg = min(max(arg, -1), 1)
        
        return acos(arg)
    
    def createCHJ(self, polygon:QPolygonF):
        """
        Create convex hull using Jarvis Scan
        """
        ch = QPolygonF()
        
        # Create pivot
        q = min(polygon, key = lambda k: k.y())
        pj = q
        
        # Create point pj1
        px = min(polygon, key = lambda k: k.x())
        pj1 = QPointF(px.x()+0.1, pj.y())
        
        
        # Add pivot to ch
        ch.append(pj)
        
        # Process all points
        while True:
            # Initialize maximum and its index
            phi_max = 0
            idx_max = -1
            
            for i in range(len(polygon)):
                # Different points
                if (pj != polygon[i]):
                    # Compute angle
                    phi = self.get2VectorsAngle(pj, pj1, pj, polygon[i])
                
                    # Update maximum
                    if phi > phi_max:
                        phi_max = phi
                        idx_max = i
                    
            # Add point with maximum angle to ch
            ch.append(polygon[idx_max])
            
            # Update indeces
            pj1 = pj
            pj = polygon[idx_max]
            
            # Condition to stop
            if pj == q:
                break
            
        return ch
    
    
    def createCHG(self, polygon:QPolygonF):
        # Initialize polygon
        ch = QPolygonF()
        
        n = len(polygon)
        
        # Create pivot
        q = min(polygon, key = lambda k: k.y())
        
        # Extra point to create vector from which the angles will be calculated
        extra_point = QPointF((q.x()+1),q.y())
        
        # Add pivot to convex hull
        ch.append(q)
        angles_dict = {}
        
        for i in range (n):
            # Iterate trough all vertices and calculate angles
            
            # Skip pivot
            if q == polygon[i]:
                continue
            
            ang = self.get2VectorsAngle(q,extra_point,q,polygon[i])
            
            # Check for points with the same angle and keep the furter one from pivot
            if ang in angles_dict.keys(): 
                prev_point_dist = self.calculateDistance(q,polygon[angles_dict[ang]])
                current_point_dist = self.calculateDistance(q,polygon[i])
                if current_point_dist < prev_point_dist:
                    continue
                
            angles_dict[ang] = i
        
        # Sort indices of veritces dpending on the angle
        sorted_indices = [angles_dict[angle] for angle in sorted(angles_dict.keys())]
        
        # Add first (non-pivot) point to convex hull
        ch.append(polygon[sorted_indices[0]])
        
        # Iterate trough sorted vertices
        for j in range(1, len(sorted_indices)):
            current_point = polygon[sorted_indices[j]]

            while len(ch) > 1 and self.get_point_location(current_point, ch[-2], ch[-1]) == -1:
                
                # Remove last point if it makes a clockwise turn
                ch.remove(len(ch)-1)
                  
            # Add current point
            ch.append(current_point)

        return ch
                
    
    
    def rotate(self, pol: QPolygonF, sigma):
        '''
        Rotate polygon by angle sigma
        '''
        pol_r = QPolygonF()
        
        # Process points one by one
        for p in pol:
            # Rotate polygon point
            x_r = p.x()*cos(sigma) - p.y()*sin(sigma)
            y_r = p.x()*sin(sigma) + p.y()*cos(sigma)
            
            # Create point
            p_r = QPointF(x_r, y_r)
            
            # Add point to polygon
            pol_r.append(p_r)
        
        return pol_r
    
    
    def createMMB(self, pol: QPolygonF):
        '''
        Create minmax box
        '''
        mmb = QPolygonF()
        
        # Find extreme coordinates
        x_min = min(pol, key = lambda k: k.x()).x()
        x_max = max(pol, key = lambda k: k.x()).x()
        
        y_min = min(pol, key = lambda k: k.y()).y()
        y_max = max(pol, key = lambda k: k.y()).y()
        
        # Compute area
        area = (x_max - x_min) * (y_max - y_min)
        
        # Create min-max box vertices
        v1 = QPointF(x_min, y_min)
        v2 = QPointF(x_max, y_min)
        v3 = QPointF(x_max, y_max)
        v4 = QPointF(x_min, y_max)
        
        # Create min-max box polygon
        mmb.append(v1)
        mmb.append(v2)
        mmb.append(v3)
        mmb.append(v4)

        return mmb, area
    
    
    def getArea(self, pol: QPolygonF):
        '''
        Compute aera of a polygon
        '''
        area = 0
        n = len(pol)
        
        # Process vertices one by one
        for i in range(n):
            area += pol[i].x() * (pol[(i+1)%n].y() - pol[(i-1+n)%n].y())
        
        return abs(area)/2
    
    
    def resizeRectangle(self, building: QPolygonF, mbr: QPolygonF):
        '''
        Resizing rectangle to match the building area
        '''
        mbr_res = QPolygonF()
        
        # Compute k
        Ab = self.getArea(building)
        A = self.getArea(mbr)
        if A != 0:
            k = Ab/A
        else:
            k = 1
        
        # Compute centroid
        x_t = 0.25*(mbr[0].x()+mbr[1].x()+mbr[2].x()+mbr[3].x())
        y_t = 0.25*(mbr[0].y()+mbr[1].y()+mbr[2].y()+mbr[3].y())
        
        # Compute vectors
        v1_x = mbr[0].x() - x_t
        v1_y = mbr[0].y() - y_t
        
        v2_x = mbr[1].x() - x_t
        v2_y = mbr[1].y() - y_t
        
        v3_x = mbr[2].x() - x_t
        v3_y = mbr[2].y() - y_t
        
        v4_x = mbr[3].x() - x_t
        v4_y = mbr[3].y() - y_t
        
        # Compute coordinates of resized points
        v1_xr = x_t + v1_x * sqrt(k)
        v1_yr = y_t + v1_y * sqrt(k)
        
        v2_xr = x_t + v2_x * sqrt(k)
        v2_yr = y_t + v2_y * sqrt(k)
        
        v3_xr = x_t + v3_x * sqrt(k)
        v3_yr = y_t + v3_y * sqrt(k)
        
        v4_xr = x_t + v4_x * sqrt(k)
        v4_yr = y_t + v4_y * sqrt(k)
        
        # Compute new vertices
        v1_res = QPointF(v1_xr, v1_yr)
        v2_res = QPointF(v2_xr, v2_yr)
        v3_res = QPointF(v3_xr, v3_yr)
        v4_res = QPointF(v4_xr, v4_yr)     
        
        # Add vertices to the resized mbr
        mbr_res.append(v1_res)
        mbr_res.append(v2_res)
        mbr_res.append(v3_res)
        mbr_res.append(v4_res)
        
        return mbr_res
    

    def createMBR(self, building: QPolygonF, method):
        '''
        Simplify building using MBR
        '''
        sigma_min = 0
        
        # Create convex hull
        if method == 1:
            ch = self.createCHJ(building)
        elif method == 2:
            ch = self.createCHG(building)
        
        # Initilize MBR and its area
        mmb_min, area_min = self.createMMB(ch)
        
        # Browse CH segments
        n = len(ch)
        for i in range(n):
            # Coordinate differences
            dx = ch[(i+1)%n].x() - ch[i].x()
            dy = ch[(i+1)%n].y() - ch[i].y()
            
            # Compute direction
            sigma = atan2(dy, dx)
            
            # Rotate by minus sigma
            ch_r = self.rotate(ch, -sigma)
            
            # Compute MMB and its area
            mmb, area = self.createMMB(ch_r)
            
            # Update minimum
            if area < area_min:
                area_min = area
                mmb_min = mmb
                sigma_min = sigma
        
        # Resize rectangle
        mmb_min_res = self.resizeRectangle(building, mmb_min)
        result = self.rotate(mmb_min_res, sigma_min)
        # Convert min-max box with the minimum area MBR
        return result,sigma_min
    
    
    def createBRPCA(self, building: QPolygonF):
        '''
        Simplify building using PCA
        '''
        x, y = [], []
           
        # Coonvert points to coordinates     
        for p in building:
            x.append(p.x())
            y.append(p.y())
        
        # Create A
        A = np.array([x, y])
        
        # Covariance matix
        C = np.cov(A)
        
        # SVD - Singular Value Decomposition
        [_, _, V] = np.linalg.svd(C)
        
        # Direction of the principal vector
        sigma = atan2(V[0][1], V[0][0])
        
        # Rotate by minus sigma
        building_r = self.rotate(building, -sigma)
        
        # Compute MMB and its area
        mmb, _ = self.createMMB(building_r)
        
        # Resize rectangle
        mmb_res = self.resizeRectangle(building,  mmb)
        
        # Convert min-max box with the minimum area MBR
        return self.rotate(mmb_res, sigma),sigma
    
    def wallAverage(self, building:QPolygonF):
        
        n = len(building)
        
        dx0 = building[1].x() - building[0].x()
        dy0 = building[1].y() - building[0].y()
        
        d_sum = 0
        rd = 0
        
        # Reference angle
        sigma0 = atan2(dy0,dx0)
        
        # Iterate rough all vertices
        for i in range (n):          
            dx = building[(i+1)%n].x() - building[i].x()
            dy = building[(i+1)%n].y() - building[i].y()
            
            # Calculate direction
            sigma_i = atan2(dy,dx)
            
            # Calculate difference between reference angle and direction
            dif_sigma = abs(sigma_i-sigma0)
            
            # Calculate modulo
            k = round(2*dif_sigma/pi)
            r = dif_sigma - k*pi/2
            
            # Calcluate length of an edge
            d = self.calculateDistance(building[(i+1)%n],building[i])
            rd += r*d
            d_sum+=d
        
        # Calculate final sigma    
        sigma = sigma0 + rd/d_sum
        
        rot = self.rotate(building,-sigma)
        mmb, _ = self.createMMB(rot)
        mmb_rot = self.rotate(mmb,sigma)
        mmb_res = self.resizeRectangle(building,mmb_rot)
        
        return mmb_res,sigma
            
                
    def longestEdge(self, building:QPolygonF):
        n = len(building)
        longest_edge = 0
        
        # Calculate lengths of all edges
        for i in range (n):
            edge_length = self.calculateDistance(building[i],building[(i+1)%n])
            
            # Find longest edge
            if edge_length > longest_edge:
                longest_edge = edge_length
                a = building[i]
                b = building[(i+1)%n]
        
        # Find sigma of the longest edge
        sigma = atan2((b.y() - a.y()), (b.x() - a.x()))
        
        rot = self.rotate(building, -sigma)
        mmb, _ = self.createMMB(rot)
        mmb_rot = self.rotate(mmb, sigma)
        mmb_res = self.resizeRectangle(building, mmb_rot)

        return mmb_res,sigma
    
    def weightedBisector(self,building:QPolygonF):
        n = len(building)
        
        # Diagonal doesn't exist
        if n < 4:
            return QPolygonF()

        potential_diagonals = {}
        
        # Find all diagonals and calculate their length
        for i in range (n-1):
            for j in range ((i+1),n):
                if abs(i-j) <= 1 or abs(i-j) > n-2:
                    continue
                
                a, b = building[i], building[j]
                diag_id = ((a.x(), a.y()), (b.x(), b.y()))
                
                length = self.calculateDistance(a,b)
                potential_diagonals[diag_id] = length
                
        # Sort diagonald by length        
        sorted_diagonals = dict(sorted(potential_diagonals.items(), key=lambda item: item[1], reverse=True))
        two_diag_len = []
        two_diag_dir = []
        
        # Check for intersections of edges and diagonals using half-plane tests, starting with longest diagonals
        for (a_coords, b_coords), length in sorted_diagonals.items():
            a, b = QPointF(*a_coords), QPointF(*b_coords)
            
            for i in range(n):
                c = building[i]
                d = building[(i+1)%n]
                
                # Half-plane tests
                ta = self.get_point_location(a,c,d)
                tb = self.get_point_location(b,c,d)
                tc = self.get_point_location(c,a,b)
                td = self.get_point_location(d,a,b)
                
                # Check for valid (non-intersecting) diagonals
                if ta == tb or tc == td or a == c or a == d or b == c or b == d:
                    if i == n-1:
                        dx = b.x() - a.x()
                        dy = b.y() - a.y()
                        
                        sigma0 = atan2(dy,dx)
                        
                        two_diag_len.append(length)
                        two_diag_dir.append(sigma0)
                        
                        # After finding two valid diagonals, find sigma and finish simplificaction
                        if len(two_diag_len) == 2:
                            sigma = (two_diag_len[0]*two_diag_dir[0]+two_diag_len[1]*two_diag_dir[1])/sum(two_diag_len)
                            rot = self.rotate(building,-sigma)
                            mmb, _ = self.createMMB(rot)
                            mmb_rot = self.rotate(mmb,sigma)
                            mmb_res = self.resizeRectangle(building,mmb_rot)
                            return mmb_res,sigma
                    else:    
                        continue
                else:
                    break
    def normalizeAngle(self,angle):
        # Normalizes angle between -pi and pi
        return (angle + pi) % (2 * pi) - pi            
                
    def evaluate(self,sigma,building:QPolygonF):
        # Finds average squared difference between edges direction and main direction
        n = len(building)
                
        ris=[]

        # Iterate rough all vertices
        for i in range (n):          
            dx = building[(i+1)%n].x() - building[i].x()
            dy = building[(i+1)%n].y() - building[i].y()
            
            # Calculate direction
            sigma_i = atan2(dy,dx)
            
            # Calculate difference between reference angle and direction
            dif_sigma = self.normalizeAngle(sigma_i-sigma)
            
            # Calculate angle remainder
            ki = (2*dif_sigma)/pi
            ri = (ki-round(ki))*(pi/2)
            ris.append(ri**2)
        
        # Calculate squared reamainders error (in degrees)
        angle_dif = (90/n)*sqrt(sum(ris))
        return angle_dif
