# Demographic inference from cost-effective DNA sequencing experiments using deep learning
Repository for Bioinformatics Research Project as a requirement for MSc Bioinformatics at Queen Mary's Univesiry of London. Working in Fumagalli lab in collaboration with Cornell Univeristy for 'American Shad' demographic information.

- [The Project Breakdown](#project-breakdown)
- [Requirements](#requirements)
- [Models](#models)
- [Summary Statistics](#stats)



# The project breakdown
<a name="project-breakdown"></a>

**Background**

As anthropogenic pressures increasingly threaten the persistence of many species, we urgently need a better understanding of how organisms respond to changing conditions. We can leverage information from genomes of multiple samples to dissect the demographic history of extant populations. Recently, deep learning algorithms have emerged as a powerful framework to infer evolutionary scenarios, including demographic history, from population genomic data (Korfmann et al. 2023). The recent revolution in DNA sequencing technology makes it possible to now generate genome-scale data for any organism. However, it is not yet clear how well deep learning algorithms work for such uncertain data.

**Objectives** 

The broad aim of this project was to design, test, and deploy a novel deep learning algorithm that accommodated sparse and uncertain low-coverage sequencing data. With this framework, we then sought to develop a proof-of-concept application by examining how well we could recover known demographic events in real data on American shad, a fish that had recently established large populations in a new geographic region. The proposed work thus addressed two specific objectives:

Designed and implemented a generative adversarial network for demographic inferences from uncertain genomic data.
Trained and deployed the network to infer parameters relating to the demographic history of American shad populations.

**Methodology**

To meet objective 1, we made use of generative adversarial networks, algorithms that comprised two neural networks: the generator to simulate data and the discriminator to distinguish between observed and synthetic data. We modified a recently proposed architecture, implemented in the software pg-gan, for demographic inference.

To address objective 2, we applied the approach developed under objective 1 to data on the American shad. This fish is natively distributed along the east coast of North America and was introduced on the west coast of North America 150 years ago where it quickly established large populations and is abundant today. This welldocumented introduction at a known time in the past makes American shad a useful test case because it allows us to examine how accurately we can infer a known demographic event (data courtesy of Cornell University)

# Requirements
<a name="requirements"></a>

- msprime version
- numpy version
- matplotlib verison
- cvs vertsion
- tqdm version
- As well as requirements following the [pg-gan github page](https://github.com/mathiesonlab/pg-gan)



# Models
<a name="Models"></a>
Discaimer! This is only how the models I have listed are made and can be used, to utilise these models in pg-gan, please check out [pg-gan github page](https://github.com/mathiesonlab/pg-gan) for further information. 

The main focus of the project is the American shad, see 'shad_simple.py' for the main msprime model. To test the ability of the model and accuracy when replicating different demographic scenarios, two other models for the Atlantic Salmon and Atlantic Striped Bass were also created. 'salmon.py' demonstrates the msprime model for a decreasing population and 'striped_bass.py' for an increasing population.

**Creating your own model**

As seen in all the models, there is a function called "add_demographic_event", this simplifies the way the user can add different events in the model for specific population sizes and events.

```
def add_demographic_event(start_size, end_size, start_time, end_time):
    rate = np.log(end_size / start_size) / (end_time - start_time)
    for t in range(start_time, end_time + 1):
        population_size = int(start_size * np.exp(rate * (t - start_time)))
        demographic_events.append(msprime.PopulationParametersChange(time=t / generation_time, initial_size=population_size))
```
Following this you can input your desired population information e.g. 

add_demographic_event(population start size, population end size, start time for event , end time for event)

Check the [msprime page](https://tskit.dev/msprime/docs/stable/intro.html) for more information on the software itself and parameters. Here is an example from 'salmon.py , lines 22,23,24:

```
add_demographic_event(2_000, 1_000, 1, 5)
    add_demographic_event(1_000, 2_000, 6, 8)
    add_demographic_event(2_000, 1_000, 9, 10)
```

Flat regions work better when specified like this:
```
    for t in range(11, 20):
        demographic_events.append(msprime.PopulationParametersChange(time=t / generation_time, initial_size=1_000))
```
**Defining parameters for the model**

To define parameters for the model change the values within the python file. Check the [msprime page](https://tskit.dev/msprime/docs/stable/intro.html) for more information on the software itself and parameters. It is also important to change the generation time in the parameters list in the fish function:
```
def fish(params, sample_sizes, length, seed, reco, generation_time=6):
....
params = {"mut": 2.0e-9}
sample_sizes = [10]
length = 100_000
seed = None
reco = 1.6e-6
generation_time = 6
num_simulations = 100
```


# Summary Statistics
<a name="stats"></a>

The summary statistics included in the model .py files are for SINGLE POPULATION analysis only. When the script is run it will produce stats for Tajima's D, Allele Frequency Spectrum and Diversity (pi), as well plotting them, and saving all these stats in a csv file, for ease of access if further analysis, or comparison is required. 

EXAMPLE

In terminal it will print:
```
Nucleotide Diversity: mean = 7.823333333333333e-05, std = 1.8802347502504522e-05
Tajima's D: mean = -0.06605912841509974, std = 0.4569053828913878
```
And then return:

1) Population graph for the model
<img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/e122a0b8-fb4e-430d-887a-5fb732616265" alt="Population Graph" width="400">

2) Nucleotide Diveristy (pi) Across Simulations
<img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/5397dacf-5abb-437b-b76c-6bd9b9bf9adb" alt="Nucleotide Diversity (pi)" width="400">

3) Tajima's D across simulations
<img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/f25bd633-9804-4a12-8f7f-31ed928d5fbf" alt="Tajima's D" width="400">

4) Allele Frequency Spectrum Across Simulations
<img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/250a1c77-0b02-492d-a74d-94f2b389fa1a" alt="Allele Frequency Spectrum" width="400">

5) Allele Frequency Spectrum Combined
<img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/d16542c1-f28b-4523-974d-405d8e3e3913" alt="Allele Frequency Spectrum Combined" width="400">
