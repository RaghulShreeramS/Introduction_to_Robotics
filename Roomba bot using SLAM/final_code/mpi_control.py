# -*- coding: utf-8 -*-
"""
Copyright 2023, UC San Diego, Contextual Robotics Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from megapi import MegaPi


MFR = 2     # port for motor front right
MBL = 3     # port for motor back left
MBR = 10    # port for motor back right
MFL = 11    # port for motor front left


class MegaPiController:
    def __init__(self, port='/dev/ttyUSB0', verbose=True):
        self.port = port
        self.verbose = verbose
        if verbose:
            self.printConfiguration()
        self.bot = MegaPi()
        self.bot.start(port=port)
        self.mfr = MFR  # port for motor front right
        self.mbl = MBL  # port for motor back left
        self.mbr = MBR  # port for motor back right
        self.mfl = MFL  # port for motor front left 
        # Constant multiplicative modifier for FL, FR, RL, RR
        # self.k = [1.00, 0.8875, 0.91458, 0.82875]   
        # self.k = [0.95, 1.00, 1.00, 0.95] # forward   
        # self.k = [0.88, 0.87, 0.925, 0.95]   
        # self.k = [ 0.93331579,  0.92278947,  0.99368421,  0.940        ]
        # self.k = [ 0.88331579,  0.92278947,  0.99368421,  0.900        ]
        # self.k = [ 0.86831579,  0.92278947,  0.99368421,  0.885        ]
        # self.k = [ 0.86031579,  0.92278947,  0.99368421,  0.885        ]
        # self.k = [ 0.86031579,  0.92778947,  0.98868421,  0.880        ]

        # self.k = [ 0.87016236, 0.9384083, 1.00, 0.89007187] # HW3, HW4
        self.k = [ 1.00, 0.9384083, 1.00, 0.89007187] # HW5 still FL wheel

    
    def printConfiguration(self):
        print('MegaPiController:')
        print("Communication Port:" + repr(self.port))
        print("Motor ports: MFR: " + repr(MFR) +
              " MBL: " + repr(MBL) + 
              " MBR: " + repr(MBR) + 
              " MFL: " + repr(MFL))


    def setFourMotors(self, vfl=0, vfr=0, vbl=0, vbr=0):
        if self.verbose:
            print("Set Motors: vfl: " + repr(int(round(vfl,0))) + 
                  " vfr: " + repr(int(round(vfr,0))) +
                  " vbl: " + repr(int(round(vbl,0))) +
                  " vbr: " + repr(int(round(vbr,0))))


        self.bot.motorRun(self.mfl,-int(round(self.k[0]*vfl)))
        self.bot.motorRun(self.mfr,int(round(self.k[1]*vfr)))
        self.bot.motorRun(self.mbl,-int(round(self.k[2]*vbl)))
        self.bot.motorRun(self.mbr,int(round(self.k[3]*vbr)))

        # self.bot.motorRun(self.mfl,-vfl)
        # self.bot.motorRun(self.mfr,vfr)
        # self.bot.motorRun(self.mbl,-vbl)
        # self.bot.motorRun(self.mbr,vbr)


    def carStop(self):
        if self.verbose:
            print("CAR STOP:")
        self.setFourMotors()


    def carStraight(self, speed):
        if self.verbose:
            print("CAR STRAIGHT:")
        self.setFourMotors(speed, speed, speed, speed)


    def carRotate(self, speed):
        if self.verbose:
            print("CAR ROTATE:")
        self.setFourMotors(-speed, speed, -speed, speed)


    def carSlide(self, speed):
        if self.verbose:
            print("CAR SLIDE:")
        self.setFourMotors(-speed, speed, speed, -speed)

    
    def carMixed(self, v_straight, v_rotate, v_slide):
        if self.verbose:
            print("CAR MIXED")
        self.setFourMotors(
            -(v_rotate-v_straight+v_slide),
            v_rotate+v_straight+v_slide,
            -(v_rotate-v_straight-v_slide),
            v_rotate+v_straight-v_slide
        )
    
    def close(self):
        self.bot.close()
        self.bot.exit()


if __name__ == "__main__":
    import time
    mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
    time.sleep(1)
    # speed = 65

    # mpi_ctrl.carStraight(50)
    # time.sleep(6)
    # mpi_ctrl.carStop()
    # time.sleep(1)
    # mpi_ctrl.carSlide(speed)
    # time.sleep(6)
    # mpi_ctrl.carRotate(speed)
    # time.sleep(4)
    #mpi_ctrl.carSlide(speed)
    #time.sleep(4)
    # mpi_ctrl.carMixed(-speed,speed,speed)
    # time.sleep(4)


    # mpi_ctrl.setFourMotors()
    mpi_ctrl.carStop()
    # print("If your program cannot be closed properly, check updated instructions in google doc.")
