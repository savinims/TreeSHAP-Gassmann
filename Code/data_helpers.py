"""
@author: Savini Samarasinghe
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.polynomial.polynomial as poly
# =============================================================================
# Common helper functions for data processing
# =============================================================================


def calculate_gassmann_ksat(phi, k_fl, k_mineral, k_dry):
    """
    args:
        phi: numpy array
            porosity (a fraction)
        k_fl: numpy array
            bulk modulus of the target fluid
        k_mineral: numpy array
            bulk modulus of the mineral - depends on the composition of the rock (e.g. percent of clay/quartz)
        k_dry: numpy array
            bulk modulus of the dry rock

    returns: 
        k_sat: numpy array
            bulk modulus of the saturated rock
    """
    denominator = phi/k_fl + (1-phi)/k_mineral - k_dry/(k_mineral**2)
    k_sat = k_dry + ((1-k_dry/k_mineral)**2)/denominator
    return k_sat


def get_bulk_modulus(rho, v_p, v_s):
    """"
    args: 
        rho: numpy array
            grain density
        v_p: numpy array
            p-wave velocity
        v_s: numpy array
            s-wave velocity

    returns: numpy array
        bulk modulus

    """
    return rho*(v_p**2 - 4*(v_s**2)/3)


def get_shear_modulus(rho, v_s):
    """"
    args: 
        rho: numpy array
            grain density
        v_s: numpy array
            s-wave velocity

    returns: numpy array
        shear modulus

    """
    return rho*(v_s**2)

def get_gassmann_velocities(k_sat_gassmann, g_dry, rho_saturated):
    """"
    args: 
        k_sat_gassmann: numpy array
            bulk modulus of the saturated rock estimated from the Gassmann's model 
        g_dry: numpy array
            shear modulus of the dry rock
        rho_saturated: numpy array
            grain density of the saturated rock


    returns: 
        v_p: numpy array
            p-wave velocity
        v_s: numpy array
            s-wave velocity
    """
    v_p = np.sqrt( (k_sat_gassmann + (4/3)*g_dry) /rho_saturated)
    v_s = np.sqrt(g_dry/rho_saturated)
    
    return v_p, v_s


def print_k_sat_error(df):
    """
    function to print the mean and standard deviation of the Gassmann apporximation error
    args:
        df: pandas DataFrame
            a dataframe containing k_sat and k_sat_gassmann information
    returns: None
    """
    squared_error = (df['k_sat']-df['k_sat_gassmann'])**2
    print('Mean of squared error = {:.4f}'.format(np.mean(squared_error)))
    print('Standard deviation of squared error = {:.4f}' .format(
        np.std(squared_error)))


def draw_cross_plot(x_data, y_data, x_label, y_label, colorbar_label='|Measured - Gassmann|', show=True, color_by_var=None, cmap='copper_r', line=True, vmin=None, vmax=None):
    """
        A custom function to draw cross plots
        args:
            x_data: numpy array
                data of the x-axis
            y_data: numpy array
                data of the y-axis
            x_label: str
                x label text
            y_label: str
                y label text
            colorbar_label: str
                colorbar label text
            show: boolean
                set to False to be able to save the figures
            color_by_var: numpy array
                color the dots in the scatter plot based on this data
            cmap: str
                name of the colormap
            line: boolean
                indicates whether the x=y line should be drawn
            vmin: float
                min value of the colormap
            vmax: float
                max value of the colormap              
    """

    min_val = np.nanmin([x_data, y_data])
    max_val = np.nanmax([x_data, y_data])
    data_range = np.arange(min_val, max_val+1)

    plt.figure(figsize=(20, 10))
    if cmap == 'seismic':
        lim = np.max(abs(color_by_var))
        plt.scatter(x_data, y_data, c=color_by_var,
                    cmap=cmap, alpha=0.8, vmin=-lim, vmax=lim)
    else:
        if not vmin:
            plt.scatter(x_data, y_data, c=color_by_var, cmap=cmap, alpha=0.8)
        else:
            plt.scatter(x_data, y_data, c=color_by_var, cmap=cmap,
                        alpha=0.8, vmin=vmin, vmax=vmax)
    plt.colorbar(label=colorbar_label)

    plt.xlabel(x_label, size=14)
    plt.ylabel(y_label, size=14)
    if line:
        plt.plot(data_range, data_range, color='black')
    plt.grid()
    if show:
        plt.show()


# =============================================================================
# Custom functions to read and process the input files
# =============================================================================

def get_blangy_dataframe(input_folder_path, rho_fluid, k_fluid, k_quartz, k_clay, k_mica, k_feldspar):

    file_list = ['/Blangy.92.T2.csv', '/Blangy.92.T3.csv', '/Blangy.92.T4.csv', '/Blangy.92.T5.csv', '/Blangy.92.T6.csv',
                 '/Blangy.92.T7.csv']
    details = pd.read_csv(input_folder_path+'/Blangy_details.csv')
    processed_df = pd.DataFrame()

    for data_file in file_list:
        df = pd.read_csv(input_folder_path+data_file)
        df = df.dropna(axis=0)  # drops rows with missing velocity values
        df = df.merge(details, on='Sample nbr', how='left')

        porosity_label = 'Corr poros'
        porosity = df[porosity_label].values

        # assuming rho_air =0
        rho_dry = (1-porosity)*df['grain density'].values
        rho_saturated = (1-porosity) * \
            df['grain density'].values + porosity*rho_fluid
        k_dry = get_bulk_modulus(
            rho_dry, df['Vp dry'].values, df['Vs dry'].values)
        k_sat = get_bulk_modulus(
            rho_saturated, df['Vp wet'].values, df['Vs wet'].values)
        g_dry = get_shear_modulus(rho_dry, df['Vs dry'].values)
        g_sat = get_shear_modulus(rho_saturated, df['Vs wet'].values)

        feldspar = df['feldspar'].values/100
        df['feldspar'] = feldspar
        mica = df['mica'].values/100
        df['mica'] = mica
        quartz = df['quartz'].values/100
        df['quartz'] = quartz
        df['clay'] = 0

        # calculate k_mineral using VRH averaging scheme
        k_mineral = ((feldspar*k_feldspar + quartz*k_quartz + mica*k_mica) + 1/(feldspar/k_feldspar +
                                                                                quartz/k_quartz + mica/k_mica))*0.5
        k_sat_gassmann = calculate_gassmann_ksat(
            porosity, np.ones_like(porosity)*k_fluid, k_mineral, k_dry)

        df['g_dry'] = g_dry
        df['k_dry'] = k_dry
        df['g_sat'] = g_sat
        df['k_sat'] = k_sat
        df['k_sat_gassmann'] = k_sat_gassmann
        df['k_mineral'] = k_mineral
        df['k_fl'] = np.ones_like(porosity)*k_fluid

        df['vp_sat_gassmann'], df['vs_sat_gassmann']   = get_gassmann_velocities(k_sat_gassmann, g_dry, rho_saturated)


        processed_df = pd.concat([processed_df, df], axis=0)

    processed_df.rename(columns={'Corr poros': 'porosity', 'Vp wet': 'vp_sat', 'Vs wet': 'vs_sat', 'Vp dry': 'vp_dry',
                                 'Vs dry': 'vs_dry', 'grain density': 'grain_density',
                                 'Eff pressure': 'pressure'}, inplace=True)

    return processed_df[['pressure', 'porosity', 'k_dry', 'k_mineral', 'grain_density', 'quartz', 'clay', 'mica', 'feldspar', 'vp_dry', 'vs_dry', 'vp_sat', 'vs_sat',
                         'g_dry', 'g_sat', 'k_sat', 'k_sat_gassmann', 'vp_sat_gassmann', 'vs_sat_gassmann']]


def get_strandenes_df(input_folder_path, rho_fluid, k_fluid, k_quartz, k_clay, k_mica, k_feldspar):

    file_list = ['/Strand.20.csv', '/Strand.25.csv',
                 '/Strand.30.csv', '/Strand.35.csv', '/Strand.40.csv']
    details1 = pd.read_csv(input_folder_path+'/Strand_details1.csv')
    details2 = pd.read_csv(input_folder_path+'/Strand_details2.csv')
    porosity_label = 'Porosity'
    processed_df = pd.DataFrame()

    for data_file in file_list:
        df = pd.read_csv(input_folder_path+data_file)
        df = df.dropna(axis=0)
        df = df.merge(details1, on='Sample nbr', how='left')
        df = df.merge(details2, on='Sample nbr', how='left')
        df['Clay content'] = df['Clay content'].fillna(0)
        df['Mica content'] = df['Mica content'].fillna(0)

        porosity = df[porosity_label].values
        rho_dry = (1-porosity)*df['grain density'].values
        rho_saturated = (1-porosity) * \
            df['grain density'].values + porosity*rho_fluid
        k_dry = get_bulk_modulus(
            rho_dry, df['Vp dry'].values, df['Vs dry'].values)
        k_sat = get_bulk_modulus(
            rho_saturated, df['Vp wet'].values, df['Vs wet'].values)
        g_dry = get_shear_modulus(rho_dry, df['Vs dry'].values)
        g_sat = get_shear_modulus(rho_saturated, df['Vs wet'].values)

        clay = df['Clay content'].values/100
        df['clay'] = clay
        mica = df['Mica content'].values/100
        df['mica'] = mica
        quartz = 1 - clay - mica  # we don't have quartz information
        df['quartz'] = quartz
        df['feldspar'] = 0

        k_mineral = ((quartz*k_quartz + mica*k_mica + clay*k_clay) +
                     1/(clay/k_clay + quartz/k_quartz + mica/k_mica))*0.5
        k_sat_gassmann = calculate_gassmann_ksat(
            porosity, np.ones_like(porosity)*k_fluid, k_mineral, k_dry)
        df['vp_sat_gassmann'], df['vs_sat_gassmann']   = get_gassmann_velocities(k_sat_gassmann, g_dry, rho_saturated)


        df['g_dry'] = g_dry
        df['k_dry'] = k_dry
        df['g_sat'] = g_sat
        df['k_sat'] = k_sat
        df['k_fl'] = np.ones_like(porosity)*k_fluid
        df['k_sat_gassmann'] = k_sat_gassmann
        df['k_mineral'] = k_mineral
        
        processed_df = pd.concat([processed_df, df], axis=0)
        
        # having only some of the  experiments conducted from multiple directions
        # can possibly inflate the importance of those samples in the study. therefore we will only select one direction. in this dataset, mjority of the measurements have been taken perpendicularly, i.e., direction = 1. 
        processed_df = processed_df.loc[processed_df['Direction']==1]

    processed_df.rename(columns={'Corr poros': 'porosity', 'Vp wet': 'vp_sat', 'Vs wet': 'vs_sat', 'Vp dry': 'vp_dry', 'Vs dry': 'vs_dry', 'grain density': 'grain_density',
                                 'Eff pressure': 'pressure',  'Porosity': 'porosity'}, inplace=True)
    

    return processed_df[['pressure', 'porosity', 'k_dry', 'k_mineral', 'grain_density', 'quartz', 'clay', 'mica', 'feldspar', 'vp_dry', 'vs_dry', 'vp_sat', 'vs_sat',
                          'g_dry', 'g_sat', 'k_sat', 'k_sat_gassmann', 'vp_sat_gassmann', 'vs_sat_gassmann']]


def get_han_df(input_folder_path, rho_fluid, k_quartz, k_clay, k_mica, k_feldspar, k_fluid):

    df = pd.read_csv(input_folder_path+'/Han.csv')
    df = df.dropna(axis=0)
    details = pd.read_csv(input_folder_path+'/Han_dw.csv')
    processed_df = pd.merge(
        df, details, on=['Clay', 'Vp_wet', 'Vs_wet', 'Porosity'])

    porosity = processed_df['Porosity'].values
    rho_saturated = processed_df['Dw'].values
    rho_dry =rho_saturated - porosity*rho_fluid
    grain_density = rho_dry/(1-porosity)

    k_dry = get_bulk_modulus(
        rho_dry, processed_df['Vp_dry'].values, processed_df['Vs_dry'].values)

    clay = processed_df['Clay'].values
    processed_df['quartz'] = 1-clay
    processed_df['mica'] = 0
    processed_df['feldspar'] = 0

    # K matrix is calculated using the empirical approach outlined in the dissertation
    k_mineral = []
    for i in range(len(clay)):
        if clay[i] > 0:
            v_p = 5.59 - 2.19*clay[i]
            v_s = 3.52 - 1.89*clay[i]
            rho_m = rho_dry[i]/(1-porosity[i])
            empirical = get_bulk_modulus(rho_m, v_p, v_s)
            k_mineral.append(empirical)
        else:
            k_mineral.append(40)

    k_mineral = np.array(k_mineral)
    k_sat_gassmann = calculate_gassmann_ksat(
        porosity, np.ones_like(porosity)*(k_fluid), k_mineral, k_dry)
    g_dry = get_shear_modulus(rho_dry, processed_df['Vs_dry'].values)
    g_sat = get_shear_modulus(rho_saturated, processed_df['Vs_wet'].values)

    processed_df['vp_sat_gassmann'], processed_df['vs_sat_gassmann']   = get_gassmann_velocities(k_sat_gassmann, g_dry, rho_saturated)

    processed_df['g_dry'] = g_dry
    processed_df['g_sat'] = g_sat
    processed_df['k_mineral'] = k_mineral
    processed_df['k_dry'] = k_dry
    processed_df['k_sat_gassmann'] = k_sat_gassmann
    processed_df['k_fl'] = np.ones_like(porosity)*(k_fluid)
    processed_df['grain_density'] = grain_density
    processed_df['Pressure'] = processed_df['Pressure'] - 1

    processed_df.rename(columns={'Vp_wet': 'vp_sat', 'Vs_wet': 'vs_sat', 'Vp_dry': 'vp_dry', 'Vs_dry': 'vs_dry',
                                 'K_sat': 'k_sat', 'Pressure': 'pressure', 'Clay': 'clay', 'Mica content': 'mica', 'Porosity': 'porosity'}, inplace=True)

    return processed_df[['pressure', 'porosity', 'k_dry', 'k_mineral', 'grain_density', 'quartz', 'clay', 'mica', 'feldspar', 'vp_dry', 'vs_dry', 'vp_sat', 'vs_sat',
                         'g_dry', 'g_sat', 'k_sat', 'k_sat_gassmann', 'vp_sat_gassmann', 'vs_sat_gassmann']]


def get_mapeli_df(input_folder_path, rho_fluid, k_quartz, k_clay, k_mica, k_feldspar, k_fluid):

    composition = pd.read_csv(input_folder_path+'/composition_cesar.csv')
    data = pd.read_csv(input_folder_path+'/Cesar_data.csv')
    data = data.dropna(axis=0)
    processed_df = pd.merge(data, composition, on=['Sample Name'])
    porosity_df = pd.read_csv(input_folder_path+'/Cesar_porosity.csv')
    processed_df = pd.merge(processed_df, porosity_df, on=[
                            'Sample Name', 'Pressure (psi)'])
    processed_df = processed_df.dropna(axis=0)

    porosity = processed_df['Porosity'].values/100
    processed_df['porosity'] = porosity

    rho_dry = (1-porosity)*processed_df['Grain Density (g/cm3)'].values
    rho_saturated = (
        1-porosity)*processed_df['Grain Density (g/cm3)'].values + porosity*rho_fluid
    k_dry = get_bulk_modulus(
        rho_dry, processed_df['Vp_dry'].values, processed_df['Vs_dry'].values)
    k_sat = get_bulk_modulus(
        rho_saturated, processed_df['Vp_wet'].values, processed_df['Vs_wet'].values)
    g_dry = get_shear_modulus(rho_dry, processed_df['Vs_dry'].values)
    g_sat = get_shear_modulus(rho_saturated, processed_df['Vs_wet'].values)

    clay = processed_df['Clay'].values
    # carbon = df['Carbon'].values # adding carbon to clay wasn't helpful
    feldspar = processed_df['Feldspar'].values
    quartz = processed_df['Quartz'].values
    # Ignored carbon in the K matrix calculation
    k_mineral = ((feldspar*k_feldspar + quartz*k_quartz + clay*k_clay) +
                 1/(feldspar/k_feldspar + quartz/k_quartz + clay/k_clay))*0.5
    k_sat_gassmann = calculate_gassmann_ksat(
        porosity, np.ones_like(porosity)*k_fluid, k_mineral, k_dry)
    processed_df['vp_sat_gassmann'], processed_df['vs_sat_gassmann']   = get_gassmann_velocities(k_sat_gassmann, g_dry, rho_saturated)
    processed_df['mica'] = 0
    processed_df['g_dry'] = g_dry
    processed_df['k_dry'] = k_dry
    processed_df['g_sat'] = g_sat
    processed_df['k_sat'] = k_sat
    processed_df['k_fl'] = np.ones_like(porosity)*k_fluid
    processed_df['k_sat_gassmann'] = k_sat_gassmann
    processed_df['k_mineral'] = k_mineral
    
    # having only some of the  experiments conducted from multiple directions
    # can possibly inflate the importance of those samples in the study. therefore we will only select one direction. in this dataset, mjority of the measurements have been taken at zero aand 45 degrees (counts: 0:74, 45:74, 90: 15)
    processed_df = processed_df.loc[processed_df['Direction']==0]
    
    processed_df.rename(columns={'Vp_wet': 'vp_sat', 'Vs_wet': 'vs_sat', 'Vp_dry': 'vp_dry', 'Vs_dry': 'vs_dry',
                                 'Pressure (Mpa)': 'pressure',  'Grain Density (g/cm3)': 'grain_density',
                                 'Quartz': 'quartz', 'Feldspar': 'feldspar', 'Clay': 'clay'}, inplace=True)

    processed_df = processed_df.drop(columns='Porosity')
    return processed_df[['pressure', 'porosity', 'k_dry', 'k_mineral', 'grain_density', 'quartz', 'clay', 'mica', 'feldspar', 'vp_dry', 'vs_dry', 'vp_sat', 'vs_sat',
                         'g_dry', 'g_sat', 'k_sat', 'k_sat_gassmann', 'vp_sat_gassmann', 'vs_sat_gassmann']]


def get_yin_df(input_folder_path, rho_fluid, k_quartz, k_clay, k_mica, k_feldspar, k_fluid):

    processed_df = pd.read_csv(input_folder_path+'/yin.csv')
    # to keep things simple, let's only take the up1 and down1 data
    processed_df = processed_df.loc[(processed_df['Load']=='up1')|(processed_df['Load']=='down1')]
    porosity_pressure_df = pd.read_csv(input_folder_path+'/yin porosity.csv')
    
    porosity_pressure_up_df = porosity_pressure_df[porosity_pressure_df['Direction'] == 'up']
    porosity_pressure_down_df = porosity_pressure_df[porosity_pressure_df['Direction'] == 'down']

    # interpolate uploading data
    poly_order = 2
    # porosity data is unavailable at these pressures. therefore we will approximate by interpolating the available data
    #x_new = np.array([2.5, 5, 7.5, 15])
    # we will take an average of the uploading and downloading values 
    # therefore we will focus on the pressures at which both up and down loading
    # data are available
    x_new = np.array([15])
    y_new = []

    # for the different clay contents
    for clay_colomn in porosity_pressure_df.columns[2:]:
        # fit a polynomial between pressure and porosity
        coefs = poly.polyfit(porosity_pressure_up_df['Pressure'].values[:4],
                             porosity_pressure_up_df[clay_colomn].values[:4], poly_order)
        # approximate for the missing pressure values
        ffit = poly.polyval(x_new, coefs)
        y_new.append(ffit)

    interpolated_up_df = pd.DataFrame(
        np.array(y_new).T, columns=porosity_pressure_df.columns[2:])
    interpolated_up_df['Pressure'] = x_new
    interpolated_up_df['Direction'] = 'up'

    appended_df_up = pd.concat([porosity_pressure_up_df, interpolated_up_df])
    appended_df_up.sort_values('Pressure')

    # interpolate downloading data
    poly_order = 2
    x_new = np.array([15, 20, 40])
    y_new = []

    for clay_colomn in porosity_pressure_df.columns[2:]:
        coefs = poly.polyfit(
            porosity_pressure_down_df['Pressure'].values, porosity_pressure_down_df[clay_colomn].values, poly_order)
        ffit = poly.polyval(x_new, coefs)
        y_new.append(ffit)

    interpolated_down_df = pd.DataFrame(
        np.array(y_new).T, columns=porosity_pressure_df.columns[2:])
    interpolated_down_df['Pressure'] = x_new
    interpolated_down_df['Direction'] = 'down'

    appended_df_down = pd.concat(
        [porosity_pressure_down_df, interpolated_down_df])
    appended_df_down.sort_values('Pressure')

    phi = []
    for i in range(len(processed_df)):
        direction = processed_df['Direction'].values[i]
        clay = str(processed_df['Clay'].values[i])
        pressure = processed_df['Pressure'].values[i]
        if (direction == 'up') & (pressure >= 10):
            phi.append(
                appended_df_up[(appended_df_up['Pressure'] == pressure)][clay].values[0])
        elif ((direction == 'down') & (pressure >= 10)):
            phi.append(
                appended_df_down[(appended_df_down['Pressure'] == pressure)][clay].values[0])
        else:
            phi.append(np.nan)

    phi = np.array(phi)
    processed_df['Porosity'] = phi
        
    # densitiy values for clay and sand from the dissertation
    grain_density = processed_df['Clay'].values/100 * \
        2.52 + (1 - processed_df['Clay'].values/100)*2.65
    processed_df['grain_density'] = grain_density
    processed_df = processed_df.dropna(axis=0)
        
    # given that we are not treating the uploading and downloading data separately in the other datasets, let's take the average
    processed_df = processed_df.groupby(['Pressure','Clay','Quartz'], as_index = False).mean()

    porosity = processed_df['Porosity'].values
    grain_density = processed_df['grain_density'].values

    rho_dry = (1-porosity)*grain_density
    rho_saturated = (1-porosity)*grain_density + porosity*rho_fluid

    k_dry = get_bulk_modulus(
        rho_dry, processed_df['Vp_dry'].values, processed_df['Vs_dry'].values)
    k_sat = get_bulk_modulus(
        rho_saturated, processed_df['Vp_sat'].values, processed_df['Vs_sat'].values)

    g_dry = get_shear_modulus(rho_dry, processed_df['Vs_dry'].values)
    g_sat = get_shear_modulus(rho_saturated, processed_df['Vs_sat'].values)

    clay = processed_df['Clay'].values/100
    quartz = processed_df['Quartz'].values/100
    processed_df['Clay'] = clay
    processed_df['Quartz'] = quartz
    processed_df['mica'] = 0
    processed_df['feldspar'] = 0

    k_mineral = ((quartz*k_quartz + clay*k_clay) + 1 /
                 (quartz/k_quartz + clay/k_clay))*0.5
    k_sat_gassmann = calculate_gassmann_ksat(
        porosity, np.ones_like(porosity)*k_fluid, k_mineral, k_dry)

    processed_df.insert(11, 'k_sat', k_sat[:, np.newaxis])
    processed_df.insert(11, 'k_sat_gassmann', k_sat_gassmann[:, np.newaxis])

    processed_df['k_dry'] = k_dry
    processed_df['g_dry'] = g_dry
    processed_df['g_sat'] = g_sat
    processed_df['k_mineral'] = k_mineral
    processed_df['vp_sat_gassmann'], processed_df['vs_sat_gassmann']   = get_gassmann_velocities(k_sat_gassmann, g_dry, rho_saturated)
    processed_df.rename(columns={'Vp_sat': 'vp_sat', 'Vs_sat': 'vs_sat', 'Vp_dry': 'vp_dry', 'Vs_dry': 'vs_dry',
                                 'Pressure': 'pressure', 'Porosity': 'porosity',
                                 'Quartz': 'quartz',  'Clay': 'clay'}, inplace=True)
    processed_df['pressure'] = processed_df['pressure'] - \
        1  # pore pressure kept constant at 1MPa
        
    
    return processed_df[['pressure', 'porosity', 'k_dry', 'k_mineral', 'grain_density', 'quartz', 'clay', 'mica', 'feldspar', 'vp_dry', 'vs_dry', 'vp_sat', 'vs_sat',
                         'g_dry', 'g_sat', 'k_sat', 'k_sat_gassmann', 'vp_sat_gassmann', 'vs_sat_gassmann']]
   