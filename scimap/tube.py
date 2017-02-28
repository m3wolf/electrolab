# -*- coding: utf-8 -*-

import math

# kalpha2 is half the intensity of kalpha1
KALPHA2_RATIO = 0.5


class XRDTube():
    def __init__(self, kalpha1, kalpha2):
        self.kalpha1 = kalpha1
        self.kalpha2 = kalpha2

    @property
    def kalpha(self):
        wavelength = (
            (self.kalpha1 + KALPHA2_RATIO * self.kalpha2) /
            (1 + KALPHA2_RATIO)
        )
        return wavelength

    def split_angle_by_kalpha(self, angle):
        """Predict kα1/kα2 splitting at the given 2θ angle."""
        theta1 = math.degrees(
            math.asin(self.kalpha1 * math.sin(math.radians(angle)) / self.kalpha)
        )
        theta2 = math.degrees(
            math.asin(self.kalpha2 * math.sin(math.radians(angle)) / self.kalpha)
        )
        return (theta1, theta2)

tubes = {
    'Cu': XRDTube(kalpha1=1.5406,
                  kalpha2=1.5444),
}
