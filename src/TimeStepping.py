class TimeStepping:
    def __init__(self,namelist):
        self.Δt = namelist['time_stepping']['dt']
        self.Δti = 1.0/self.Δt

        self.Δt_up = self.Δt
        self.Δti_up = 1.0/self.Δt_up

        self.t_max = namelist['time_stepping']['t_max']
        self.t = 0.0
        self.nstep = 0
        return

    def update(self):
        self.t += self.Δt
        self.nstep += 1
        return
