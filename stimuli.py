#
# updated from ian's original code
#   include ability to label individual stimuli (1,2,3,4)
#   change size to 256x256 for resnet training
#

import math
import numpy as np
import skimage.draw

class Figure5:

    SIZE = (100, 100)
    RANGE = (1, 90) #sizes of angles generated
    POS_RANGE = (20, 80) #position range
    AUTO_SPOT_SIZE = 3 #how big automatic spot is in pixels
    LENGTH_MAX = 34 #how long a line can be for length
    LENGTH_MIN = 5 #lines disappear when they're too short
    ANGLE_LINE_LENGTH = 12 #how long the angle lines are
    AREA_DOF = 12 #maximum circle radius
    VOLUME_SIDE_MAX = 14
    CURV_DOF = 34
    CURV_WIDTH = 22 #auto curvature width
    WIGGLE = 5 #How many pixels of x "wiggling" it can do

    @staticmethod
    def calc_ranges(stimulus): #Calculates what the range of values is that "flags" supplies as a preset to other methods.
        R = 0 #Highest number that can be generated (range: 1-R)
        if stimulus is Figure5.angle:
            R = Figure5.RANGE[1]
        elif stimulus is Figure5.length:
            R = Figure5.LENGTH_MAX - Figure5.LENGTH_MIN + 1
        elif stimulus is Figure5.direction:
            R = 360
        elif stimulus is Figure5.area:
            R = Figure5.AREA_DOF
        elif stimulus is Figure5.volume:
            R = Figure5.VOLUME_SIDE_MAX
        elif stimulus is Figure5.curvature:
            R = Figure5.CURV_DOF
        elif stimulus is Figure5.position_non_aligned_scale or stimulus is Figure5.position_common_scale:
            R = Figure5.POS_RANGE[1] - Figure5.POS_RANGE[0] + 1
        return R 

    @staticmethod #for run_regression_isvetkey.py
    def _min(stimulus):
        if stimulus is Figure5.length:
            return Figure5.LENGTH_MIN
        elif stimulus is Figure5.position_non_aligned_scale or stimulus is Figure5.position_common_scale:
            return Figure5.POS_RANGE[0]
        return 1

    @staticmethod #for run_regression_isvetkey.py
    def _max(stimulus):
        if stimulus is Figure5.length:
            return Figure5.LENGTH_MAX
        elif stimulus is Figure5.position_non_aligned_scale or stimulus is Figure5.position_common_scale:
            return Figure5.POS_RANGE[1]
        return Figure5.calc_ranges(stimulus)


    @staticmethod #Driver method
    def flags(stimulus, flags): #flag 1: diagonal vs random, flag 2: x wiggle, flag 3: which is largest?
        sparse_ = [] #sparse of all stimuli
        label_ = [] #label of all stimuli
        parameters = 1 #number of permutations for all 4 combined
        X = 15
        XR = X
        if flags[1]:
            XR = X + np.random.randint((-1)*Figure5.WIGGLE, Figure5.WIGGLE)
            parameters *= (Figure5.WIGGLE*2+1)
        if flags[0]:
            Y = 20
        elif stimulus is not Figure5.position_common_scale and stimulus is not Figure5.position_non_aligned_scale:
            Y = np.random.randint(Figure5.POS_RANGE[0], Figure5.POS_RANGE[1])
            parameters *= (Figure5.POS_RANGE[1] - Figure5.POS_RANGE[0] + 1)

        R = Figure5.calc_ranges(stimulus)
        parameters *= (R**4)
        sizes = [0] * 4
        for i in range(len(sizes)):
            sizes[i] = np.random.randint(1, R)
            if stimulus is Figure5.position_non_aligned_scale or stimulus is Figure5.position_common_scale:
                sizes[i] = sizes[i] + Figure5.POS_RANGE[0] - 1 #Fixes 1-61 to become 20-80
            elif stimulus is Figure5.length:
                sizes[i] += Figure5.LENGTH_MIN - 1
        if flags[2]:
            L = sizes[0]
            SL = 0
            for i in range(1, len(sizes)):
                if sizes[i] > L:
                    L = sizes[i]
                    SL = i
            if SL > 0:
                sizes[0], sizes[SL] = sizes[SL], sizes[0]
            parameters = parameters / 4

        if stimulus is Figure5.position_non_aligned_scale:
            diff = np.random.randint(-9, 11)
            parameters *= 21
            temp = stimulus(X=XR, preset=sizes[0], recur=True, diff=diff)
        elif stimulus is Figure5.position_common_scale:
            temp = stimulus(X=XR, preset=sizes[0], recur=True)
        else:
            temp = stimulus(X=XR, Y=Y, preset=sizes[0], recur=True, label=1)
        img = temp[1]
        sparse_.append(temp[0])
        label_.append(temp[2])
        if len(temp) > 3:
            parameters *= temp[3]

        for i in range(1, 4):
            X = XR = X + 22
            if flags[1]:
                XR = X + np.random.randint((-1)*Figure5.WIGGLE, Figure5.WIGGLE)
                parameters *= (Figure5.WIGGLE*2+1)
            if flags[0]:
                Y = Y + 20
            elif stimulus is not Figure5.position_common_scale and stimulus is not Figure5.position_non_aligned_scale:
                Y = np.random.randint(Figure5.POS_RANGE[0], Figure5.POS_RANGE[1])
                parameters *= (Figure5.POS_RANGE[1] - Figure5.POS_RANGE[0] + 1)
            if stimulus is Figure5.position_non_aligned_scale:
                temp = stimulus(X=XR, preset=sizes[i], preset_img=img, recur=True, diff=diff)
            elif stimulus is Figure5.position_common_scale:
                temp = stimulus(X=XR, preset=sizes[i], preset_img=img, recur=True)
            else:
                temp = stimulus(X=XR, Y=Y, preset=sizes[i], preset_img=img, recur=True, label=i+1)
            sparse_.append(temp[0])
            label_.append(temp[2])
            if len(temp) > 3:
                parameters *= temp[3]
            #if parameters was an odd number (like 61^4) divided by 4, make it whole
            parameters = int(parameters)
        return sparse_, img, label_, parameters
    

    #FOR ALL STIMULUS METHODS
    #flags = the flags passed by the user (explanation in flags method)
    #X, Y = position of stimulus on image
    #preset = whatever is being randomized (direction, length, radius, etc)
    #preset_img = when it's the 2nd-4th stimulus being added to an existing image
    #recur = is it the user's call (false) or is it the flags method calling it (true)
    #Use of variable "recur" allows me to use methods as their own helper methods

    @staticmethod
    def position_non_aligned_scale(flags=[False, False, False], X=0, preset=None, preset_img=None, recur=False, diff=None, varspot=False):
        if not recur:
            return Figure5.flags(Figure5.position_non_aligned_scale, flags)
        if preset_img is not None:
            img = preset_img
        else:
            img = np.zeros(Figure5.SIZE)
            ORIGIN = 7 #where the line is
            if diff is not None:
                img[Figure5.POS_RANGE[0]+diff:Figure5.POS_RANGE[1]+diff, ORIGIN] = 1
            else:
                diff = np.random.randint(-9, 11)
                img[Figure5.POS_RANGE[0]+diff:Figure5.POS_RANGE[1]+diff, ORIGIN] = 1
        parameters = 1
        if varspot:
            sizes = [1, 3, 5, 7, 9, 11]
            spot_size = np.random.choice(sizes)
            parameters *= len(sizes)
        else:
            spot_size = Figure5.AUTO_SPOT_SIZE
        if preset is None:
            Y = np.random.randint(Figure5.POS_RANGE[0]+diff, Figure5.POS_RANGE[1]+diff)
        else:
            Y = preset
        Y = Y + diff
        label = Y - Figure5.POS_RANGE[0] - diff
        
        half_size = spot_size / 2
        img[int(Y-half_size):int(Y+half_size+1), int(X-half_size):int(X+half_size+1)] = 1

        sparse = [Y, X, spot_size]

        return sparse, img, label, parameters

    @staticmethod
    def position_common_scale(flags=[False, False, False], X=0, preset=None, preset_img=None, recur=False, varspot=False):
        if not recur:
            return Figure5.flags(Figure5.position_common_scale, flags)
        return Figure5.position_non_aligned_scale(X=X, preset=preset, preset_img=preset_img, varspot=varspot, recur=recur, diff=0)
    
    @staticmethod
    def angle(flags=[False, False, False], X=0, Y=0, preset=None, preset_img=None, recur=False, label=1) :
        if not recur:
            return Figure5.flags(Figure5.angle, flags)
        if preset_img is not None:
            img = preset_img
        else:
            img = np.zeros(Figure5.SIZE)
        L = Figure5.ANGLE_LINE_LENGTH
        startangle = np.random.randint(0, 359)
        parameters = 360
        if preset is None:
            ANGLE = np.random.randint(Figure5.RANGE[0], Figure5.RANGE[1])
        else:
            ANGLE = preset
        t2 = startangle * (math.pi/180)
        diff2 = ANGLE * (math.pi/180)
        r, c = skimage.draw.line(Y, X, Y+(int)(L*np.sin(t2)), X+(int)(L*np.cos(t2)))
        diffangle = t2+diff2 #angle after diff is added (2nd leg)
        r2, c2 = skimage.draw.line(Y, X, Y+(int)(L*np.sin(diffangle)), X+(int)(L*np.cos(diffangle)))
        img[r, c] = label
        img[r2, c2] = label
        sparse = {"y": Y,"x": X,"angle": ANGLE,"startangle": startangle}
        return sparse, img, ANGLE, parameters

    @staticmethod
    def length(flags=[False, False, False], X=0, Y=0, preset=None, preset_img=None, recur=False, label=1):
        if not recur:
            return Figure5.flags(Figure5.length, flags)
        if preset_img is not None:
            img = preset_img
        else:
            img = np.zeros(Figure5.SIZE)
        if preset is None:
            L = np.random.randint(1, Figure5.LENGTH_MAX)
        else:
            L = preset
        half_l = int(L * 0.5)
        img[Y-half_l:Y+half_l, X] = 1
        sparse = [Y, X, L]
        return sparse, img, L

    @staticmethod
    def direction(flags=[False, False, False], X=0, Y=0, preset=None, preset_img=None, recur=False, label=1):
        if not recur:
            return Figure5.flags(Figure5.direction, flags)
        if preset_img is not None:
            img = preset_img
        else:
            img = np.zeros(Figure5.SIZE)
        L = Figure5.ANGLE_LINE_LENGTH
        if preset is None:
            angle = np.random.randint(0, 360)
        else:
            angle = preset
        radangle = angle * np.pi / 180
        r, c = skimage.draw.line(Y, X, Y+int(L*np.sin(radangle)), X+int(L*np.cos(radangle)))
        img[r,c] = 1
        img[Y-1:Y+1, X-1:X+1] = 1
        sparse = [Y, X, angle]
        return sparse, img, angle

    @staticmethod
    def area(flags=[False, False, False], X=0, Y=0, preset=None, preset_img=None, recur=False, label=1):
        if not recur:
            return Figure5.flags(Figure5.area, flags)
        if preset_img is not None:
            img = preset_img
        else:
            img = np.zeros(Figure5.SIZE)
        if preset is None:
            radius = np.random.randint(1, Figure5.AREA_DOF+1)
        else:
            radius = preset
        r, c = skimage.draw.ellipse_perimeter(Y, X, radius, radius)
        img[r, c] = 1
        sparse = [Y, X, radius]
        label = np.pi * radius * radius
        return sparse, img, label

    @staticmethod
    def volume(flags=[False, False, False], X=0, Y=0, preset=None, preset_img=None, recur=False, label=1):
        if not recur:
            return Figure5.flags(Figure5.volume, flags)
        if preset_img is not None:
            img = preset_img
        else:
            img = np.zeros(Figure5.SIZE)

        if preset is None:
            depth = np.random.randint(1, Figure5.VOLUME_SIDE_MAX)
        else:
            depth = preset

        def obliqueProjection(point):
            angle = -45.
            alpha = (np.pi/180.0) * angle
            P = [[1, 0, (1/2.)*np.sin(alpha)], [0, 1, (1/2.)*np.cos(alpha)], [0, 0, 0]]
            ss = np.dot(P, point)
            return [int(np.round(ss[0])), int(np.round(ss[1]))]
        halfdepth = int(depth/2.)
        fbl = (Y+halfdepth, X-halfdepth)
        fbr = (fbl[0], fbl[1]+depth)
    
        r, c = skimage.draw.line(fbl[0], fbl[1], fbr[0], fbr[1])
        img[r, c] = 1

        ftl = (fbl[0]-depth, fbl[1])
        ftr = (fbr[0]-depth, fbr[1])
        
        r, c = skimage.draw.line(ftl[0], ftl[1], ftr[0], ftr[1])
        img[r,c] = 1
        r, c = skimage.draw.line(ftl[0], ftl[1], fbl[0], fbl[1])
        img[r, c] = 1
        r, c = skimage.draw.line(ftr[0], ftr[1], fbr[0], fbr[1])
        img[r, c] = 1

        bbr = obliqueProjection([fbr[0], fbr[1], depth])
        btr = (bbr[0]-depth, bbr[1])
        btl = (btr[0], btr[1]-depth)

        r, c = skimage.draw.line(fbr[0], fbr[1], bbr[0], bbr[1])
        img[r, c] = 1
        r, c = skimage.draw.line(bbr[0], bbr[1], btr[0], btr[1])
        img[r, c] = 1
        r, c = skimage.draw.line(btr[0], btr[1], btl[0], btl[1])
        img[r, c] = 1
        r, c = skimage.draw.line(btl[0], btl[1], ftl[0], ftl[1])
        img[r, c] = 1
        r, c = skimage.draw.line(btr[0], btr[1], ftr[0], ftr[1])
        img[r, c] = 1

        sparse = [Y, X, depth]
        label = depth ** 3
        return sparse, img, label
        

    @staticmethod
    def curvature(flags=[False, False, False], X=0, Y=0, preset=None, preset_img=None, recur=False, varwidth=False, label=1):
        if not recur:
            return Figure5.flags(Figure5.curvature, flags)
        if preset_img is not None:
            img = preset_img
        else:
            img = np.zeros(Figure5.SIZE)
        if preset is None:
            depth = np.random.randint(1, Figure5.CURV_DOF)
        else:
            depth = preset
        width = Figure5.CURV_WIDTH
        parameters = 1
        halfwidth = int(width/2)
        if varwidth:
            width = np.random.randint(1, halfwidth)*2
            parameters = halfwidth
        start = (Y, X-halfwidth)
        mid = (Y-depth, X)
        end = (Y, X+halfwidth)
        r, c = skimage.draw.bezier_curve(start[0], start[1], mid[0], mid[1], end[0], end[1], 1)
        img[r, c] = 1
        sparse = [Y, X, depth, width]
        t = 0.5
        P10 = (mid[0] - start[0], mid[1] - start[1])
        P21 = (end[0] - mid[0], end[1] - mid[1])
        dBt_x = 2*(1-t)*P10[1] + 2*t*P21[1]
        dBt_y = 2*(1-t)*P10[0] + 2*t*P21[0]
        dBt2_x = 2*(end[1] - 2*mid[1] + start[1])
        dBt2_y = 2*(end[0] - 2*mid[0] + start[0])
        curvature = np.abs((dBt_x*dBt2_y - dBt_y*dBt2_x) / ((dBt_x**2 + dBt_y**2)**(3/2.)))
        label = np.round(curvature, 3)
        return sparse, img, label, parameters
