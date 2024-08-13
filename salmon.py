import msprime
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm  # Import norm from scipy.stats for p-value calculation

#msprime model
def fish(params, sample_sizes, length, seed, reco, generation_time=6):
    assert len(sample_sizes) == 1

    demographic_events = []
    
    #initial population size
    demographic_events.append(msprime.PopulationParametersChange(time=0, initial_size=2_000, growth_rate=0))

    def add_demographic_event(start_size, end_size, start_time, end_time):
        rate = np.log(end_size / start_size) / (end_time - start_time)
        for t in range(start_time, end_time + 1):
            population_size = int(start_size * np.exp(rate * (t - start_time)))
            demographic_events.append(msprime.PopulationParametersChange(time=t / generation_time, initial_size=population_size))

    add_demographic_event(2_000, 1_000, 1, 5)
    add_demographic_event(1_000, 2_000, 6, 8)
    add_demographic_event(2_000, 1_000, 9, 10)

    for t in range(11, 20):
        demographic_events.append(msprime.PopulationParametersChange(time=t / generation_time, initial_size=1_000))

    add_demographic_event(1_000, 5_000, 21, 30)
    add_demographic_event(5_000, 3_000, 31, 33)
    add_demographic_event(3_000, 7_000, 34, 36)
    add_demographic_event(7_000, 4_000, 37, 40)
    add_demographic_event(4_000, 8_000, 41, 45)

    for t in range(46, 50):
        demographic_events.append(msprime.PopulationParametersChange(time=t / generation_time, initial_size=8_000))

    ts = msprime.simulate(
        sample_size=sum(sample_sizes),
        demographic_events=demographic_events,
        mutation_rate=params.get("mut"),
        length=length,
        recombination_rate=reco,
        random_seed=seed
    )

    return ts, demographic_events

def wattersons_theta(ts, sample_size):
    S = ts.num_sites
    a1 = sum(1 / i for i in range(1, sample_size))
    theta_w = S / a1
    return theta_w

def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Simulation', 'Diversity', 'Tajimas_D', 'Wattersons_Theta', 'SNPs', 'Allele_Frequency_Spectrum'])
        for i, result in enumerate(results):
            writer.writerow([i+1, result[2], result[3], result[4], result[5], result[6]])

def repeat_simulations(params, sample_sizes, length, reco, generation_time, num_simulations, seed=None):
    results = []
    for i in range(num_simulations):
        print(f"Running simulation {i+1}")
        if seed is not None:
            np.random.seed(seed + i)  
        ts, demographic_events = fish(params, sample_sizes, length, seed, reco, generation_time)
        diversity = ts.diversity()
        tajimas_d = ts.Tajimas_D()
        theta_w = wattersons_theta(ts, sample_size=sum(sample_sizes))
        num_snps = ts.num_sites
        allele_frequency_spectrum = ts.allele_frequency_spectrum(polarised=True)
        results.append((ts, demographic_events, diversity, tajimas_d, theta_w, num_snps, allele_frequency_spectrum))
    
    save_results_to_csv(results, 'shad_results.csv')

    return results
#params
params = {"mut": 1.0e-8}
sample_sizes = [10]
length = 100_000
seed = None
reco = 1.6e-6
generation_time = 6
num_simulations = 200

results = repeat_simulations(params, sample_sizes, length, reco, generation_time, num_simulations, seed=seed)

#population changes
times = []
sizes = []
for event in results[0][1]:
    times.append(event.time * generation_time)
    sizes.append(event.initial_size)

times = np.array(times)
sizes = np.array(sizes)

#population plot
plt.plot(times, sizes)
plt.xlabel("Time (years)")
plt.ylabel("Population Size")
plt.title("Population Size Changes Over Time")
plt.gca().invert_xaxis()
plt.show()

# Plot summary statistics 
diversities = [result[2] for result in results]
tajimas_ds = [result[3] for result in results]
wattersons_thetas = [result[4] for result in results]
num_snps_list = [result[5] for result in results]
allele_frequency_spectra = [result[6] for result in results]

#nucleotide diversity plot
plt.figure(figsize=(10, 5))
plt.hist(diversities, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel("Nucleotide Diversity (π)")
plt.ylabel("Frequency")
plt.title("Histogram of Nucleotide Diversity Across Simulations")
plt.show()

#tajimas d plot
plt.figure(figsize=(10, 5))
plt.hist(tajimas_ds, bins=10, color='pink', edgecolor='black', alpha=0.7)
plt.xlabel("Tajima's D")
plt.ylabel("Frequency")
plt.title("Distribution of Tajima's D Across Simulations")
plt.show()

#wattersons theta plot
plt.figure(figsize=(10, 5))
plt.hist(wattersons_thetas, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
plt.xlabel("Watterson's Theta")
plt.ylabel("Frequency")
plt.title("Distribution of Watterson's Theta Across Simulations")
plt.show()

#number of snps plot
plt.figure(figsize=(10, 5))
plt.hist(num_snps_list, bins=10, color='lightcoral', edgecolor='black', alpha=0.7)
plt.xlabel("Number of SNPs")
plt.ylabel("Frequency")
plt.title("Distribution of Number of SNPs Across Simulations")
plt.show()

# Joint distribution plot
plt.figure(figsize=(10, 5))
plt.scatter(wattersons_thetas, num_snps_list, alpha=0.5)
plt.xlabel("Watterson's Theta")
plt.ylabel("Number of SNPs")
plt.title("Joint Distribution of Watterson's Theta and Number of SNPs")
plt.show()

#afs
plt.figure(figsize=(10, 5))
bar_width = 0.8 / num_simulations 
colors = plt.cm.tab20(np.linspace(0, 1, num_simulations)) 
for i, afs in enumerate(allele_frequency_spectra):
    x_positions = np.arange(len(afs)) + i * bar_width  
    plt.bar(x_positions, afs, width=bar_width, color=colors[i], label=f'Simulation {i+1}')

plt.xlabel("Frequency")
plt.ylabel("Number of Sites")
plt.title("Allele Frequency Spectrum Across Simulations")
plt.show()

#combined afs
combined_afs = np.sum(allele_frequency_spectra, axis=0)
normalized_afs = combined_afs / np.sum(combined_afs)

plt.figure(figsize=(10, 5))
plt.bar(range(len(normalized_afs)), normalized_afs, color='green', edgecolor='black', alpha=0.7)
plt.xlabel("Derived Allele Frequency")
plt.ylabel("Proportion of Sites")
plt.title("Allele Frequency Spectrum Across Simulations")
plt.show()

# Summary stats
print(f"Nucleotide Diversity: mean = {np.mean(diversities)}, std = {np.std(diversities)}")
print(f"Tajima's D: mean = {np.mean(tajimas_ds)}, std = {np.std(tajimas_ds)}")
print(f"Watterson's Theta: mean = {np.mean(wattersons_thetas)}, std = {np.std(wattersons_thetas)}")
print(f"Number of SNPs: mean = {np.mean(num_snps_list)}, std = {np.std(num_snps_list)}")
