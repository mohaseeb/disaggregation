import math
import random

import numpy as np


class SignalGenerator:
    def __init__(self):
        pass

    def generate_random_signals(self, start, end, samples_sec, signals_number):
        signals = np.zeros((signals_number, samples_sec))
        for i in range(signals_number):
            signals[i, :] = self.generate_random_signal(start, end, samples_sec)
        return signals

    def generate_random_signal(self, start, end, samples_sec):
        pass


class SineSignalGenerator(SignalGenerator):
    def __init__(self, A_range, f_range, p_range):
        SignalGenerator.__init__(self)
        self.A_range = A_range
        self.f_range = f_range
        self.p_range = p_range

    @staticmethod
    def _generate_unit_sine(f, t, p):
        return np.sin([math.radians(angle_degree) for angle_degree in 2 * 180 * f * t + p])

    def generate_signal(self, A, f, p, start, end, samples_sec):
        samples_number = (end - start) * samples_sec
        t = np.linspace(start, end, samples_number)
        return A * self._generate_unit_sine(f, t, p)

    def generate_random_signal(self, start, end, samples_sec):
        return self.generate_signal(random.choice(self.A_range), random.choice(self.f_range),
                                    random.choice(self.p_range),
                                    start, end, samples_sec)


class SquareSignalGenerator(SineSignalGenerator):
    def __init__(self, A_range, f_range, p_range):
        SineSignalGenerator.__init__(self, A_range, f_range, p_range)

    @staticmethod
    def _generate_unit_sine(f, t, p):
        return np.sign(np.sin([math.radians(angle_degree) for angle_degree in 2 * 180 * f * t + p]))
