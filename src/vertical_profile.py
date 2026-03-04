"""
Class for vertical quantum well profile construction
"""

from dataclasses import dataclass
import numpy as np

import constants

@dataclass 
class VerticalProfile():

    interface_type: str # One of 'sharp', 'linear-wall', 'smoothed-linear-wall', 'sigmoid'
    bot_cap_si_concentration: float = 0.7
    top_cap_si_concentration: float = 0.7
    well_si_concentration: float = 1.0
    well_width_nm: float = 10.0
    top_cap_width_nm: float = 10.0
    bot_cap_width_nm: float = 10.0
    top_interface_width_nm: float = 1.0
    bot_interface_width_nm: float = 1.0

    wiggle_amplitude: float = 0.0
    wiggle_period_nm: float = 1.6

    interface_type_list = ['sharp', 'linear-wall', 'smoothed-linear-wall', 'sigmoid', 'wiggle-well']

    def __post_init__(self):

        if self.top_cap_si_concentration > 1.0 or self.top_cap_si_concentration < 0.0:
            raise ValueError("Please specify a valid top_cap_si_concentration")
        if self.bot_cap_si_concentration > 1.0 or self.bot_cap_si_concentration < 0.0:
            raise ValueError("Please specify a valid bot_cap_si_concentration")
        if self.well_si_concentration > 1.0 or self.well_si_concentration < 0.0:
            raise ValueError("Please specify a valid well_si_concentration")

        # Get dimensions in terms of atomic monolayers
        self.top_cap_width_ml = self.top_cap_width_nm / (1e9 * constants.SI_LATTICE_CONSTANT/4)
        self.bot_cap_width_ml = self.bot_cap_width_nm / (1e9 * constants.SI_LATTICE_CONSTANT/4)
        self.well_width_ml = self.well_width_nm / (1e9 * constants.SI_LATTICE_CONSTANT/4)

        self.top_interface_width_ml = self.top_interface_width_nm / (1e9 * constants.SI_LATTICE_CONSTANT/4)
        self.bot_interface_width_ml = self.bot_interface_width_nm / (1e9 * constants.SI_LATTICE_CONSTANT/4)

        if self.interface_type not in self.interface_type_list:
            raise ValueError('Please specify a valid well type')
        
        match self.interface_type:
            case 'sharp':
                # Perfect quantum wells: single-monolayer interface widths

                # Define interface width of 1 ML for the perfectly sharp case
                self.top_interface_width_ml = 1 
                self.bot_interface_width_nm = 1

                # For sharp wells, all dimensions are integer numbers of monolayers
                self.top_cap_width_ml = int(np.ceil(self.top_cap_width_ml))
                self.bot_cap_width_ml = int(np.ceil(self.bot_cap_width_ml))
                self.well_width_ml = int(np.ceil(self.well_width_ml))

                self.n_layers = self.top_cap_width_ml + self.bot_cap_width_ml + self.well_width_ml

                si_concentrations = np.zeros(self.n_layers)
                si_concentrations[:self.top_cap_width_ml] = self.top_cap_si_concentration
                si_concentrations[self.top_cap_width_ml:self.top_cap_width_ml+self.well_width_ml] = self.well_si_concentration
                si_concentrations[self.top_cap_width_ml+self.well_width_ml:] = self.bot_cap_si_concentration

                self.si_concentrations = si_concentrations

            case 'sigmoid':
                # Conventional sigmoid interfaces

                self.n_layers = int(np.ceil(self.top_cap_width_ml + self.well_width_ml + self.bot_cap_width_ml))

                x0 = self.top_cap_width_ml 
                x1 = self.top_cap_width_ml + self.well_width_ml 
                tau_0 = self.top_interface_width_ml/4
                tau_1 = self.bot_interface_width_ml/4


                layers_arr = np.array(range(self.n_layers))
                ge_concentrations = np.zeros(self.n_layers)
                ge_concentrations += (self.well_si_concentration - self.top_cap_si_concentration) * 1 / (1 + np.exp( (layers_arr-x0)/tau_0 ))
                ge_concentrations += (self.well_si_concentration - self.bot_cap_si_concentration) * 1 / (1 + np.exp( (x1-layers_arr)/tau_1 ))
                ge_concentrations += (1 - self.well_si_concentration)
                self.si_concentrations = 1 - ge_concentrations


            case 'linear-wall' | 'smoothed-linear-wall':
                # Linear interfaces between the well and barrier regions
                
                # For these wells, the interfaces and top/bottom caps contribute to the profile separately.
                self.n_layers = int(np.ceil(self.top_cap_width_ml + self.top_interface_width_ml + self.well_width_ml + self.bot_interface_width_ml + self.bot_cap_width_ml))

                si_concentrations = np.zeros(self.n_layers)

                for i in range(self.n_layers):
                    if i < self.top_cap_width_ml:
                        si_concentrations[i] = self.top_cap_si_concentration

                    elif i < self.top_cap_width_ml + self.top_interface_width_ml:
                        si_concentrations[i] = self.top_cap_si_concentration + (i-self.top_cap_width_ml)*(self.well_si_concentration-self.top_cap_si_concentration)/self.top_interface_width_ml

                    elif i >= self.n_layers - self.bot_cap_width_ml:
                        si_concentrations[i] = self.bot_cap_si_concentration
  
                    elif i >= self.n_layers-self.bot_cap_width_ml - self.bot_interface_width_ml:
                        si_concentrations[i] = self.bot_cap_si_concentration + (self.n_layers-self.bot_cap_width_ml-i)*(self.well_si_concentration - self.bot_cap_si_concentration)/self.bot_interface_width_ml
                    
                    else:
                        si_concentrations[i] = self.well_si_concentration

                
                if self.interface_type == 'smoothed-linear-wall':
                    # Smooth the linear interfaces slightly
                    smoothed_si = si_concentrations.copy()
                    for i in range(1, self.n_layers-1):
                        smoothed_si[i] = (si_concentrations[i-1] + si_concentrations[i] + si_concentrations[i+1]) / 3
                    
                    si_concentrations = smoothed_si
                
                self.si_concentrations = si_concentrations


            case _:
                raise ValueError('Please specify a valid well type')
            
        self.z_arr = np.arange(self.n_layers) * constants.SI_LATTICE_CONSTANT/4
        layers_arr = np.arange(self.n_layers)

        wiggle_period_ml = self.wiggle_period_nm / (1e9 * constants.SI_LATTICE_CONSTANT/4)

        if self.wiggle_amplitude > 0.0:
            # Add wiggle to the profile
            if self.interface_type == 'sigmoid':
                # These well boundaries tend to look good
                x0 = self.top_cap_width_ml + (3/4) * self.top_interface_width_ml
                x1 = self.top_cap_width_ml + self.well_width_ml - (1/4) * self.bot_interface_width_ml

            
            elif self.interface_type in ['linear-wall', 'smoothed-linear-wall']:
                x0 = self.top_cap_width_ml + self.top_interface_width_ml
                x1 = self.n_layers - self.bot_cap_width_ml - self.bot_interface_width_ml
            
            elif self.interface_type == 'sharp':
                x0 = self.top_cap_width_ml
                x1 = self.top_cap_width_ml + self.well_width_ml

            # Only start a new oscillation if we can complete it within the well region
            # x0 and x1 are in monolayers
            x1_ = np.floor((x1-x0) / (wiggle_period_ml)) * wiggle_period_ml + x0
            
            for i in range(self.n_layers):
                if i >= x0 and i < x1_:
                    self.si_concentrations[i] -= self.wiggle_amplitude * (1/2) * (1 - np.cos(2 * np.pi * 1e9 * constants.SI_LATTICE_CONSTANT / 4 * (layers_arr[i] - x0) / self.wiggle_period_nm))
            
        self.ge_concentrations = 1 - self.si_concentrations
        
        

        
      








