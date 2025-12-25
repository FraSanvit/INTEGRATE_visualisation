import pandas as pd
import matplotlib.pyplot as plt


#######################################################################################
# CONFIGURATION SWITCHES
#######################################################################################

# Set to True to include autarky scenarios in electricity generation plot, False to exclude them
INCLUDE_AUTARKY = False


#######################################################################################
#MARKO Result Visualisation
#######################################################################################

#import makro data from csv file
macro_df=pd.read_csv("data//figure6_makro_main.csv", sep=';')

#make a bar plot for every variable in every scenario
variables = macro_df['variable'].unique()
scenarios = macro_df['scenario'].unique()

#define scenario colors
scenario_colors = {
    'TECH-unlimited': '#D3BA68',
    'TECH-limited': '#D5695D',
    'LED-unlimited': '#5D8CA8',
    'LED-limited': '#65A479'
}


########## figure 6: COMBINED MACRO FIGURES #################
# Create figure with 2 rows and 1 column for both macro plots
fig_macro, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(20, 8))

########## TOP SUBPLOT: GDP AND CONSUMPTION IN DIFFERENT SCENARIOS #################
#chose variables for the integrate plot
variables_macro=['Gross Domestic Product (real)', 'Public Consumption (real)', 'Private Consumption (real)']

# Create subplots within the top subplot
ax_macro = [plt.subplot(2, 1, 1)] * len(variables_macro)  # Placeholder, will be replaced

# Close the temporary figure and create proper nested subplots
plt.close()

fig_macro = plt.figure(figsize=(20, 10))
gs = fig_macro.add_gridspec(2, 1, hspace=0.35, left=0.08, right=0.98, top=0.96, bottom=0.08)

# Top row: 3 subplots for GDP and consumption
gs_top = gs[0].subgridspec(1, 3, wspace=0.35)
ax_macro = [fig_macro.add_subplot(gs_top[i]) for i in range(3)]

for i, variable in enumerate(variables_macro):
    macro_variable = macro_df[macro_df['variable'] == variable]
    
    # Get the reference value from TECH-unlimited scenario
    reference_data = macro_variable[macro_variable['scenario'] == 'TECH-unlimited']
    reference_value = reference_data['value'].values[0]
    
    # Get the x-axis positions for the scenarios
    scenario_positions = {scenario: idx for idx, scenario in enumerate(scenarios)}
    
    for scenario in scenarios:
        scenario_data = macro_variable[macro_variable['scenario'] == scenario]
        value = scenario_data['value'].values[0]
        
        #get scenario color from scenario_colors dictionary
        color = scenario_colors.get(scenario)
        
        #plot bars
        ax_macro[i].bar(scenario, value, color=color)
        
        # Calculate percentage difference from TECH-unlimited
        pct_diff = ((value - reference_value) / reference_value) * 100
        
        # Only add arrow and text if the value is different from reference
        if value != reference_value:
            # Add double-headed arrow between reference line and bar top
            x_pos = scenario_positions[scenario]
            arrow_offset = 0.1  # horizontal offset for the arrow (increased to avoid overlap)
            ax_macro[i].annotate('', xy=(x_pos + arrow_offset, value), 
                          xytext=(x_pos + arrow_offset, reference_value),
                          arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
            
            # Display percentage next to the arrow
            y_mid = (value + reference_value) / 2
            ax_macro[i].text(x_pos + arrow_offset + 0.12, y_mid, f'{pct_diff:.1f}%', 
                       ha='left', va='center', fontsize=14, color='k')
        
    #draw a horizontal line at the reference value (TECH-unlimited)    
    ax_macro[i].axhline(y=reference_value, color='k', linestyle='--', linewidth=1.2)
    ax_macro[i].set_title(f'{variable}', fontsize=16, pad=10)
    # Only add y-label to the first (leftmost) subplot
    if i == 0:
        ax_macro[i].set_ylabel('Value relative to benchmark', fontsize=15)
    ax_macro[i].tick_params(axis='both', labelsize=14)
    
    # Add line breaks to scenario labels
    scenario_labels = [s.replace('-', '-\n') for s in scenarios]
    ax_macro[i].set_xticklabels(scenario_labels)
    
    # Set y-limits based on the data range
    max_value = macro_variable['value'].max()
    ax_macro[i].set_ylim(bottom=1.5, top=max_value + max_value * 0.05)

# Add a) label to the top subplot area (positioned to the left of all subplots)
ax_macro[0].text(-0.25, 0.98, 'a)', transform=ax_macro[0].transAxes, fontsize=20, va='top', ha='left')

########## BOTTOM SUBPLOT: Difference to 1 for PL, PK, and CPIs ################
# Use the actual variable names from the CSV
variables_diff=['Wage Rate', 'Capital Rent', 'CPI Households', 'CPI Government']
# Shortened titles for display (same as variable names now)
variables_diff_titles = ['Wage Rate', 'Capital Rent', 'CPI Households', 'CPI Government']

# Bottom row: 4 subplots for price indices
gs_bottom = gs[1].subgridspec(1, 4, wspace=0.35)
ax_diff = [fig_macro.add_subplot(gs_bottom[i]) for i in range(4)]

for i, variable in enumerate(variables_diff):
    macro_variable = macro_df[macro_df['variable'] == variable]
    
    for scenario in scenarios:
        scenario_data = macro_variable[macro_variable['scenario'] == scenario]
        value = scenario_data['value'].values[0]
        
        #get scenario color from scenario_colors dictionary
        color = scenario_colors.get(scenario)
        
        #plot bars showing actual values (bars start from 1)
        deviation = value - 1
        ax_diff[i].bar(scenario, deviation, bottom=1, color=color)
        
        # Display the deviation as percentage
        deviation_pct = deviation * 100
        text_offset = 0.0003
        
        # Position text above for positive deviation, below for negative
        if deviation >= 0:
            ax_diff[i].text(scenario, value + text_offset, f'{deviation_pct:.1f}%', 
                       ha='center', va='bottom', fontsize=14, color='k')
        else:
            ax_diff[i].text(scenario, value - text_offset, f'{deviation_pct:.1f}%', 
                       ha='center', va='top', fontsize=14, color='k')
        
    #draw a horizontal line at 1 (the baseline)
    ax_diff[i].axhline(y=1, color='k', linestyle='--', linewidth=1.2)
    ax_diff[i].set_title(f'{variables_diff_titles[i]}', fontsize=16, pad=10)
    # Only add y-label to the first (leftmost) subplot
    if i == 0:
        ax_diff[i].set_ylabel('Deviation from benchmark', fontsize=15)
    ax_diff[i].tick_params(axis='both', labelsize=14)
    
    # Add line breaks to scenario labels
    scenario_labels = [s.replace('-', '-\n') for s in scenarios]
    ax_diff[i].set_xticklabels(scenario_labels)
    
    ax_diff[i].set_ylim(bottom=0.8, top=1.12)

# Add b) label to the bottom subplot area (positioned to the left of all subplots)
ax_diff[0].text(-0.25, 0.98, 'b)', transform=ax_diff[0].transAxes, fontsize=20, va='top', ha='left')

plt.tight_layout()
plt.show()

########## Figure 7: Distributional Results ################

#read in distributional results
distribution_df=pd.read_csv("data//figure7_distributional.csv", sep=';')

# Remove rows with NaN values
distribution_df = distribution_df.dropna()

# Filter for the specific variable
distribution_df = distribution_df[distribution_df['variable'] == 'Private Real Consumption']

#make a bar plot for every variable in every scenario and income quartile
scenarios_dist = distribution_df['scenario'].unique()
income_quartiles = distribution_df['household'].unique()

fig, ax = plt.subplots(1, len(scenarios_dist), figsize=(16, 8), sharey=True)

#plot for every scenario
for i, scenario in enumerate(scenarios_dist):
    scenario_data = distribution_df[distribution_df['scenario'] == scenario]
    
    for quartile in income_quartiles:
        quartile_data = scenario_data[scenario_data['household'] == quartile]
        if len(quartile_data) > 0:
            value = quartile_data['value'].values[0]
            
            #plot bars
            ax[i].bar(quartile, value, color=scenario_colors.get(scenario))       
    ax[i].set_title(f'{scenario}', fontsize=16)
    ax[i].set_ylim(bottom=0.9*distribution_df['value'].min(), top=1.025*distribution_df['value'].max())
    ax[i].tick_params(axis='both', labelsize=14)

    # Only set y-label on the first (leftmost) subplot
    if i == 0:
        ax[i].set_ylabel('Private Real Consumption (relative to benchmark)', fontsize=15)

plt.tight_layout()
plt.show()


########## Figure 4: Energy Supply Mix (Electricity and Heat) ################

#read in energy generation mix results
energy_supply_df=pd.read_csv("data//figure_4a_energy_supply_electricity.csv", sep=';')
heating_supply_df=pd.read_csv("data//figure_4b_energy_supply_heat.csv", sep=';')


# Define color scheme for energy sources (stronger, more vibrant colors)
energy_colors = {
    'Hydro': '#3498DB',        # Bright blue
    'Wind': '#27AE60',         # Green
    'Biomass CHP': '#E67E22',  # Orange
    'PV': '#F1C40F',           # Yellow
    'Net imports': '#9B59B6',  # Purple
    'Gas': '#E74C3C',          # Red
    'Nuclear': '#95A5A6',      # Gray
    'Coal': '#34495E'          # Dark blue-gray
}

# Define vibrant color scheme for heating sources (different from electricity)
heating_colors = {
    'Biomass boiler': "#FA6D10",           # Dark orange
    'Biomethane boiler': '#16A085',        # Teal
    'District heating biomass': '#C0392B', # Dark red
    'District heating geothermal': '#8E44AD', # Purple
    'Heat pump': '#2980B9',                # Blue
    'Solar thermal': '#E91E63'             # Pink/Magenta
}

# Create figure based on INCLUDE_AUTARKY setting
if INCLUDE_AUTARKY:
    # Show only electricity figure when autarky is included
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
else:
    # Show both electricity and heating figures when autarky is excluded
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

######################### LEFT SUBPLOT: ELECTRICITY #########################
# Filter out autarky scenarios based on configuration switch
if INCLUDE_AUTARKY:
    energy_supply_df_filtered = energy_supply_df
else:
    energy_supply_df_filtered = energy_supply_df[~energy_supply_df['scenario'].str.contains('autarky', case=False)]

scenarios_energy = energy_supply_df_filtered['scenario'].unique()
energy_sources = energy_supply_df_filtered['technolgoy'].unique()

# Prepare data for stacked bar chart with all scenarios
x_positions = range(len(scenarios_energy))

# Separate stacking for positive and negative values
bottom_positive = [0] * len(scenarios_energy)
bottom_negative = [0] * len(scenarios_energy)

for source in energy_sources:
    values = []
    for scenario in scenarios_energy:
        source_data = energy_supply_df_filtered[(energy_supply_df_filtered['scenario'] == scenario) & 
                                       (energy_supply_df_filtered['technolgoy'] == source)]
        if len(source_data) > 0:
            values.append(source_data['value'].values[0])
        else:
            values.append(0)
    
    # Get color for this source, default to a pastel gray if not defined
    color = energy_colors.get(source, '#D3D3D3')
    
    # Separate positive and negative values
    positive_values = [max(0, v) for v in values]
    negative_values = [min(0, v) for v in values]
    
    # Plot positive values stacked upwards
    if any(v > 0 for v in values):
        ax1.bar(x_positions, positive_values, width=0.6, bottom=bottom_positive, label=source, color=color)
        
        # Add text labels inside the bars for positive values
        for i, (val, bottom) in enumerate(zip(positive_values, bottom_positive)):
            if val > 0:  # Only show label if there's a value
                y_pos = bottom + val / 2  # Center of the bar segment
                ax1.text(i, y_pos, f'{val:.0f}', ha='center', va='center', 
                       fontsize=14, color='k')
        
        bottom_positive = [bottom_positive[i] + positive_values[i] for i in range(len(positive_values))]
    
    # Plot negative values stacked downwards
    if any(v < 0 for v in values):
        ax1.bar(x_positions, negative_values, width=0.6, bottom=bottom_negative, 
               label=source if not any(v > 0 for v in values) else None, color=color)
        
        # Add text labels inside the bars for negative values
        for i, (val, bottom) in enumerate(zip(negative_values, bottom_negative)):
            if val < 0:  # Only show label if there's a value
                y_pos = bottom + val / 2  # Center of the bar segment
                ax1.text(i, y_pos, f'{val:.0f}', ha='center', va='center', 
                       fontsize=14, color='k')
        
        bottom_negative = [bottom_negative[i] + negative_values[i] for i in range(len(negative_values))]

# Add a horizontal line at y=0
ax1.axhline(y=0, color='k', linewidth=0.8)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(scenarios_energy, fontsize=14)
ax1.set_xlim(-0.5, len(scenarios_energy) - 0.5)
ax1.set_ylabel('Electricity Generation (TWh/a)', fontsize=16)
ax1.set_title('Electricity Generation Mix', fontsize=18)
# Adjust legend font size based on whether autarky scenarios are included
legend_fontsize = 12 if INCLUDE_AUTARKY else 14
ax1.legend(loc='upper right', fontsize=legend_fontsize)
ax1.set_ylim(bottom=min(bottom_negative)*1.5, top=max(bottom_positive)*1.1)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
# Adjust subplot label based on whether we're showing both plots
subplot_label = '' if INCLUDE_AUTARKY else 'a)'
if subplot_label:
    ax1.text(0.02, 0.98, subplot_label, transform=ax1.transAxes, fontsize=20, va='top')
ax1.tick_params(axis='both', labelsize=14)

# Only show heating subplot when autarky scenarios are excluded
if not INCLUDE_AUTARKY:
    ######################### RIGHT SUBPLOT: HEATING #########################
    scenarios_heating = heating_supply_df['scenario'].unique()
    heating_sources = heating_supply_df['technolgoy'].unique()
    x_positions_heat = range(len(scenarios_heating))
    bottom_positive_heat = [0] * len(scenarios_heating)
    bottom_negative_heat = [0] * len(scenarios_heating)

    for source in heating_sources:
        values_heat = []
        for scenario in scenarios_heating:
            source_data_heat = heating_supply_df[(heating_supply_df['scenario'] == scenario) & 
                                           (heating_supply_df['technolgoy'] == source)]
            if len(source_data_heat) > 0:
                values_heat.append(source_data_heat['value'].values[0])
            else:
                values_heat.append(0)
        
        color_heat = heating_colors.get(source, '#95A5A6')
        positive_values_heat = [max(0, v) for v in values_heat]            
        negative_values_heat = [min(0, v) for v in values_heat]  
        
        if any(v > 0 for v in values_heat):
            ax2.bar(x_positions_heat, positive_values_heat, width=0.6, bottom=bottom_positive_heat, label=source, color=color_heat)
            for i, (val, bottom) in enumerate(zip(positive_values_heat, bottom_positive_heat)):
                if val > 0:
                    y_pos = bottom + val / 2
                    ax2.text(i, y_pos, f'{val:.0f}', ha='center', va='center', 
                           fontsize=14, color='k')
            bottom_positive_heat = [bottom_positive_heat[i] + positive_values_heat[i] for i in range(len(positive_values_heat))]
        
        if any(v < 0 for v in values_heat):
            ax2.bar(x_positions_heat, negative_values_heat, width=0.6, bottom=bottom_negative_heat, 
                   label=source if not any(v > 0 for v in values_heat) else None, color=color_heat)
            for i, (val, bottom) in enumerate(zip(negative_values_heat, bottom_negative_heat)):
                if val < 0:
                    y_pos = bottom + val / 2
                    ax2.text(i, y_pos, f'{val:.0f}', ha='center', va='center', 
                           fontsize=14, color='k')
            bottom_negative_heat = [bottom_negative_heat[i] + negative_values_heat[i] for i in range(len(negative_values_heat))]

    ax2.axhline(y=0, color='k', linewidth=0.8)
    ax2.set_xticks(x_positions_heat)
    ax2.set_xticklabels(scenarios_heating, fontsize=14)
    # Set wider x-limits to match the bar width of electricity plot (which has 4 scenarios vs heating's 2)
    ax2.set_xlim(-1.0, len(scenarios_heating))
    ax2.set_ylabel('Heat Generation (TWh/a)', fontsize=16)
    ax2.set_title('Heating Mix', fontsize=18)
    ax2.legend(loc='upper right', fontsize=14)
    ax2.set_ylim(top=max(bottom_positive_heat)*1.1 if bottom_positive_heat else 100)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.text(0.02, 0.98, 'b)', transform=ax2.transAxes, fontsize=20, va='top', ha='left')
    ax2.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.show()

########## Figure A1: Synfuel Supply Mix ################

#read in energy syn fuel mix results
energy_synfuel_df=pd.read_csv("data//Figure_a1_energy_supply_synfuel.csv", sep=';')

# Define color scheme for synfuel sources with more distinct colors
synfuel_colors = {
    'biofuel-domestic': '#27AE60',        # Green
    'methanol-domestic': '#E74C3C',       # Red
    'biomethane-domestic': '#F39C12',     # Orange/Yellow
    'hydrogen-domestic': '#9B59B6',       # Purple
    'diesel-domestic': '#34495E',         # Dark gray
    'kerosene-domestic': '#E67E22',       # Orange
    'methane-domestic': '#1ABC9C',        # Turquoise
    'biomethane-net imports': '#D4AC0D',  # Gold/Yellow
    'hydrogen-net imports': '#2E86C1',    # Blue
    'biofuel-net imports': '#196F3D',     # Dark green
    'methanol-net imports': '#943126',    # Dark red/brown
    'diesel-net imports': '#717D7E',      # Gray
    'kerosene-net imports': '#DC7633',    # Burnt orange
    'methane-net imports': '#117A65'      # Dark teal
}

# Create figure for synfuel
fig_synfuel, ax_synfuel = plt.subplots(1, 1, figsize=(12, 8))

# Filter for the four main scenarios only
main_scenarios = ['TECH-unlimited', 'TECH-limited', 'LED-unlimited', 'LED-limited']
synfuel_df_filtered = energy_synfuel_df[energy_synfuel_df['Scenario'].isin(main_scenarios)]

scenarios_synfuel = synfuel_df_filtered['Scenario'].unique()
synfuel_sources = synfuel_df_filtered['Technology'].unique()

# Prepare data for stacked bar chart
x_positions_synfuel = range(len(scenarios_synfuel))

# Separate stacking for positive and negative values
bottom_positive_synfuel = [0] * len(scenarios_synfuel)
bottom_negative_synfuel = [0] * len(scenarios_synfuel)

for source in synfuel_sources:
    values_synfuel = []
    for scenario in scenarios_synfuel:
        source_data = synfuel_df_filtered[(synfuel_df_filtered['Scenario'] == scenario) & 
                                          (synfuel_df_filtered['Technology'] == source)]
        if len(source_data) > 0:
            values_synfuel.append(source_data['Value'].values[0])
        else:
            values_synfuel.append(0)
    
    # Get color for this source
    color_synfuel = synfuel_colors.get(source, '#95A5A6')
    
    # Separate positive and negative values
    positive_values_synfuel = [max(0, v) for v in values_synfuel]
    negative_values_synfuel = [min(0, v) for v in values_synfuel]
    
    # Plot positive values stacked upwards
    if any(v > 0 for v in values_synfuel):
        ax_synfuel.bar(x_positions_synfuel, positive_values_synfuel, width=0.5, bottom=bottom_positive_synfuel, 
                       label=source, color=color_synfuel)
        
        bottom_positive_synfuel = [bottom_positive_synfuel[i] + positive_values_synfuel[i] 
                                   for i in range(len(positive_values_synfuel))]
    
    # Plot negative values stacked downwards
    if any(v < 0 for v in values_synfuel):
        ax_synfuel.bar(x_positions_synfuel, negative_values_synfuel, width=0.5, bottom=bottom_negative_synfuel, 
                      label=source if not any(v > 0 for v in values_synfuel) else None, 
                      color=color_synfuel)
        
        bottom_negative_synfuel = [bottom_negative_synfuel[i] + negative_values_synfuel[i] 
                                   for i in range(len(negative_values_synfuel))]

# Add a horizontal line at y=0
ax_synfuel.axhline(y=0, color='k', linewidth=0.8)
ax_synfuel.set_xticks(x_positions_synfuel)
ax_synfuel.set_xticklabels(scenarios_synfuel, fontsize=14)
ax_synfuel.set_ylabel('Synfuel Supply (TWh/a)', fontsize=16)
ax_synfuel.set_title('Synfuel Supply Mix', fontsize=18)
ax_synfuel.legend(loc='upper right', fontsize=8)
ax_synfuel.set_ylim(bottom=min(bottom_negative_synfuel)*1.5 if min(bottom_negative_synfuel) < 0 else -5, 
                    top=max(bottom_positive_synfuel)*1.1)
ax_synfuel.grid(axis='y', linestyle='--', alpha=0.4)
ax_synfuel.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.show()

