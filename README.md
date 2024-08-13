# Demographic inference from cost-effective DNA sequencing experiments using deep learning
Repository for Bioinformatics Research Project as a requirement for MSc Bioinformatics at Queen Mary's Univesiry of London. Working in the Fumagalli lab at QMUL in collaboration with Cornell Univeristy, as part of [Cornell’s Global Hubs initiative and Queen Mary University’s international strategy](https://www.qmul.ac.uk/media/news/2024/pr/five-research-projects-funded-as-part-of-queen-mary-partnership-with-cornell-university-.html). My project is a segment of the overall larger project between the two Universities.

- [The Project Breakdown](#project-breakdown)
- [Requirements](#requirements)
- [Models](#models)
- [Summary Statistics](#stats)
- [Machine Learning](#ml)



# The project breakdown
<a name="project-breakdown"></a>

**Background**

As anthropogenic pressures increasingly threaten the persistence of many species, we urgently need a better understanding of how organisms respond to changing conditions. We can leverage information from genomes of multiple samples to dissect the demographic history of extant populations. Recently, deep learning algorithms have emerged as a powerful framework to infer evolutionary scenarios, including demographic history, from population genomic data. The recent revolution in DNA sequencing technology makes it possible to now generate genome-scale data for any organism. However, it is not yet clear how well deep learning algorithms work for such uncertain data.

**Objectives** 

The broad aim of this project was to design, test, and deploy a novel deep learning algorithm that accommodated sparse and uncertain low-coverage sequencing data. With this framework, we then sought to develop a proof-of-concept application by examining how well we could recover known demographic events in real data on American shad, a fish that had recently established large populations in a new geographic region. The proposed work thus addressed two specific objectives:

Design and implement an algorithm and network for demographic inference from uncertain genomic data. 
Train and deploy the network to infer parameters relating to the demographic history of American shad populations.

**Methodology**

To meet objective 1, I made use of Random Forests and Neural Networks to infer. Later in the larger project (QMUL x Cornell) a recently proposed architecture will be modified, implemented in the software pg-gan, for demographic inference.
To address objective 2, I applied the approach developed under objective 1 to data on the American shad. This fish is natively distributed along the east coast of North America and was introduced on the west coast of North America 150 years ago where it quickly established large populations and is abundant today. This welldocumented introduction at a known time in the past makes American shad a useful test case because it allows us to examine how accurately we can infer a known demographic event (data courtesy of Cornell University).

# Requirements
<a name="requirements"></a>

msprime==0.7.4
numpy==1.17.2
tensorflow==2.2.0
scikit-learn>=1.0.2
matplotlib>=3.5.3
pandas>=1.3.5
seaborn>=0.12.2

- As well as requirements following the [pg-gan github page](https://github.com/mathiesonlab/pg-gan)



# Models
<a name="Models"></a>
Discaimer! This is only how the models I have listed are made and can be used, to utilise these models in pg-gan, please check out [pg-gan github page](https://github.com/mathiesonlab/pg-gan) for further information. 

The main focus of the project is the American shad, see 'shad.py' for this msprime model. To test the ability of the model and accuracy when replicating different demographic scenarios, two other models were also created. 'salmon.py' demonstrates a decreasing population and 'striped_bass.py' demonstrates an increasing population.

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

Check the [msprime page](https://tskit.dev/msprime/docs/stable/intro.html) for more information on the software itself and parameters. Here is an example from 'salmon.py':

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

To define parameters for the model, change the values within the python file. Check the [msprime page](https://tskit.dev/msprime/docs/stable/intro.html) for more information on the software itself and parameters. It is also important to change the 'generation time' in the fish function to what it is for the species you are looking at:
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

The summary statistics included in the model .py files are for SINGLE POPULATIONs only. When the model.py scripts are run, they will produce the following summary stats: nucleotide diveristy (pi), tajimas D, Wattersons theta, number of snps and allele frequency spectra. The script will print the stats in the command line, plot them and also save the values to a csv file for ease if further analysis or comparison is required.


EXAMPLE

In terminal it will print:
```
Nucleotide Diversity: mean = 7.823333333333333e-05, std = 1.8802347502504522e-05
Tajima's D: mean = -0.06605912841509974, std = 0.4569053828913878
```
And then return the graphs, for example:

<table>
  <tr>
    <td>1) Population graph for the model</td>
    <td>2) Nucleotide Diversity (pi) Across Simulations</td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/e122a0b8-fb4e-430d-887a-5fb732616265" alt="Population Graph" width="400">
    </td>
    <td>
      <img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/5397dacf-5abb-437b-b76c-6bd9b9bf9adb" alt="Nucleotide Diversity (pi)" width="400">
    </td>
  </tr>
  <tr>
    <td>3) Tajima's D across simulations</td>
    <td>4) Allele Frequency Spectrum Across Simulations</td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/f25bd633-9804-4a12-8f7f-31ed928d5fbf" alt="Tajima's D" width="400">
    </td>
    <td>
      <img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/250a1c77-0b02-492d-a74d-94f2b389fa1a" alt="Allele Frequency Spectrum" width="400">
    </td>
  </tr>
  <tr>
    <td colspan="2">5) Allele Frequency Spectrum Combined</td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="https://github.com/CeriHarman/MSc_Project/assets/63057694/d16542c1-f28b-4523-974d-405d8e3e3913" alt="Allele Frequency Spectrum Combined" width="400">
    </td>
  </tr>
</table>


# Machine Learning
<a name="ml"></a>

**the models**

To address the main aim of the project and to demonstrate the abilities of deep learning, it was important to use a simple machine learning method e.g. Random Forest or SVM and then a Neural Network. This is so we can compare the abilities of current methods and see if deep learning can offer us more. As well as the three species of fish (shad, bass, salmon) I created three basic models to represent a decreasing, increasing and stable population, in order to provide values train the Random Forest. 
Here are the values from the three basic scenarios, and how they are used in the random_forest.py:


```
def generate_synthetic_data(num_samples, scenario):
    data = []
    for _ in range(num_samples):
        if scenario == "increasing":
            diversity = np.random.normal(loc=0.000095, scale=0.000020)
            tajimas_d = np.random.normal(loc=0.0012, scale=0.36)
            wattersons_theta = np.random.normal(loc=9.5, scale=1.9)
            num_snps = np.random.normal(loc=26.8, scale=5.4)

        elif scenario == "decreasing":
            diversity = np.random.normal(loc=0.00032, scale=0.00004)
            tajimas_d = np.random.normal(loc=0.03, scale=0.20)
            wattersons_theta = np.random.normal(loc=31.7, scale=3.7)
            num_snps = np.random.normal(loc=89.8, scale=10.4)

        elif scenario == "stable":
            diversity = np.random.normal(loc=0.00040, scale=0.00004)
            tajimas_d = np.random.normal(loc=-0.003, scale=0.19)
            wattersons_theta = np.random.normal(loc=40.3, scale=3.6)
            num_snps = np.random.normal(loc=114.0, scale=10.3)
            
        afs = np.random.dirichlet(np.ones(5))
        feature_vector = [diversity, tajimas_d, wattersons_theta, num_snps] + list(afs)
        data.append(feature_vector)
    return data

   ```

With random_forest.py, you can input your summary stats, and based on the data it is trained off, it will return a prediction of the demographic event occured/ type of population is e.g. 'increasing', 'decreasing', 'stable', as well as return a confusion matrix. To input the summary stats, just change the 'new_sample_values' in the random_forest.py file :

```
new_sample_values = [
    3.91488888888889e-05,   #nucelotide diversity
    0.023968560194865148,   #tajimas d
    3.9095525319119093,     #wattersons theta 
    11.06,                  #number of snps
    0.2, 0.2, 0.2, 0.2, 0.2 #afs
]
```

**Deep Learning**

For the deep learning, I used a Neural Network. This file 'NN.py' generates synthetic data with the same values from the random forest (for comparative reasons, change if necessary for your test) and uses it to train the MLP model (sequential). This neural network has 7 layers (4 dense and 3 dropout), this was done as it helped improve the Loss Vs Epochs in validation and training. This might be different for your data so change accordingly.

The MLP:
```
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
```

The NN.py file will return the popualtion prediction, classification report, a confusion matrix and a graph to show Loss vs Epochs in validation and training:

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/6e3543bd-4b27-4613-8fe0-268cc1044604" alt="confusion_matrix" width="400"/>
    <img src="https://github.com/user-attachments/assets/bd46ffaf-a514-4cb6-ad1a-b315ab267e03" alt="loss_v_epoch" width="400"/>
</div>

