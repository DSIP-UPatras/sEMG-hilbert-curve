# Hilbert sEMG data scanning for hand gesture recognition based on Deep Learning
Accompaniment code for the manuscript 'Hilbert sEMG data scanning for hand gesture recognition based on Deep Learning' published in Springer Neural Computing and Applications (NCAA) [DOI: 10.1007/s00521-020-05128-7](www.doi.org/10.1007/s00521-020-05128-7).

## Abstract
Deep learning has transformed the field of data analysis by dramatically improving the state of the art in various classification and prediction tasks, especially in the area of computer vision.
In biomedical engineering, a lot of new work is directed toward surface electromyography (sEMG)-based gesture
recognition, often addressed as an image classification problem using convolutional neural networks (CNNs).
In this paper, we utilize the Hilbert space-filling curve for the generation of image representations of sEMG signals, which allows the application of typical image processing pipelines such as CNNs on sequence data.
The proposed method is evaluated on different state-of-the-art network architectures and yields a significant classification improvement over the approach without the Hilbert curve.
Additionally, we develop a new network architecture (MSHilbNet) that takes advantage of multiple scales of an initial Hilbert curve representation and achieves equal performance with fewer convolutional layers.

![MSHilbNet](imgs/mshilbnet.png)


## Code Dependecies
The following python packages are needed to run the code:
- tensorflow
- sklearn
- scipy
- pandas
- numpy
- matplotlib

## Usage
To replicate the experiments described in the paper run: `bash run.sh`.
Before running the code, the Ninapro-DB1 dataset should be downloaded from [link](http://ninaweb.hevs.ch/).

## License
If this code helps your research, please cite the [paper]().

```
@article{Tsinganos2020,
author = {Tsinganos, Panagiotis and Cornelis, Bruno and Cornelis, Jan and Jansen, Bart and Skodras, Athanassios},
doi = {10.1007/s00521-020-05128-7},
journal = {Neural Computing and Applications},
month = {jul},
number = {},
pages = {},
volume = {},
publisher = {Springer},
title = {Hilbert sEMG data scanning for hand gesture recognition based on Deep Learning},
year = {2020}
}
```

### Acknowledgements
The work is supported by the "Andreas Mentzelopoulos Scholarships for the University of Patras" and the VUB-UPatras International Joint Research Group on ICT (JICT).

### Contact Details
Panagiotis Tsinganos | PhD Candidate  
University of Patras, Greece  
Vrije Universiteit Brussel, Belgium  
<panagiotis.tsinganos@ece.upatras.gr>
