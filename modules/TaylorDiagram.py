#!/usr/bin/env python

__version__ = "Time-stamp: <2010-12-27 17:42 yannick@lyopc469>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

"""
Taylor diagram (Taylor, 2001) test implementation.

http://www-pcmdi.llnl.gov/about/staff/Taylor/CV/Taylor_diagram_primer.htm
"""

import numpy as NP
import matplotlib.patches as mpatches
import math

class TaylorDiagram(object):
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    """

    def __init__(self, refstd):
        """standar deviation of the reference data."""

        self.ref = refstd

    def setup_axes(self, fig, rect=111):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes.

        Wouldn't the ideal be to define its own non-linear
        transformation, so that coordinates are directly r=stddev and
        theta=correlation? I guess it would allow 
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = NP.concatenate((NP.arange(10)/10.,[0.95,0.99]))
        tlocs = NP.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str,rlocs))))

        ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=(0,NP.pi/2, # 1st quadrant
                                                     0,1.6*self.ref),
                                           grid_locator1=gl1,
                                           tick_formatter1=tf1,
                                           )

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom") # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")   # "Y axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)         # Useless
        
        # Grid
        ax.grid()

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        print("Reference std:", self.ref)
        self.ax.plot([0],self.ref,'ko', label='_')
        t = NP.linspace(0,NP.pi/2)
        r = NP.zeros_like(t) + self.ref
        self.ax.plot(t,r,'k--', label='_')


        # Draw centered RMS lines in steps of 0.5
        #ax.text(self.ref, self.ref, 'RMS', color='r', rotation=45)
        pivot = int(3.0*self.ref)
        influence_radius = NP.arange(pivot/10.,pivot,pivot/10.)
        print(influence_radius)
        for i in influence_radius:
            i = math.ceil(i)
            c = mpatches.Circle((self.ref, 0), i, fc="none", ls='dashed', ec="grey", lw=0.5)
            ax.add_patch(c)
            if i <= 3.1*pivot/10.:
                	val = "%.1f" %(i)
                	ax.text(self.ref+0.259*float(i), 0.966*float(i), val, color='grey', horizontalalignment='center', verticalalignment='center')
        return self.ax

    def get_coords(self, sampleR, sampleStd):
        """Computes theta=arccos(correlation),rad=stddev of sample
        wrt. reference sample."""

        std = sampleStd
        theta = NP.arccos(sampleR)

        return theta,std

    def plot_sample(self, sampleR, sampleStd, *args, **kwargs):
        """Add sample to the Taylor diagram. args and kwargs are
        directly propagated to the plot command."""

        t,r = self.get_coords(sampleR, sampleStd)
        l, = self.ax.plot(t,r, *args, **kwargs) # (theta,radius)

        return l



