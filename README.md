# Data Assimilation Project 

Reconstruction based on sparse and noisy observations for Lorenz 63 and Lorenz 96 physical states with Data Assimilation and Neural Networks

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General info
This project was made during the Advanced Course on Deep Learning and Geophsyical Dynamics. This course was co-organized by AI Chairs OceaniX (https://cia-oceanix.github.io/) and AI4Child.
	
## Technologies
Project is created with:
* Python : 3.7.7
* Pytorch Lightning : 1.5.2 <br/>
PyTorch Lightning is a PyTorch framework for building neural networks. Its documentation is  <a href="https://pytorch-lightning.readthedocs.io/en/latest/"> here</a>.
	
## Setup
To run this project, install it locally :

Install PyTorch Lightning with ` pip install pytorch_lightning ` 

Make sure the `utils.py` is in the same folder as your different Python Notebook.

## Features
Four notebooks are ready to use : 
- `Baseline_L63.ipynb`: 4DVar and CNN implemented in Pytorch Lightning for Lorenz 63.
- `Baseline_L96.ipynb`: 4DVar and CNN implemented in Pytorch Lightning for Lorenz 96.
- `4DVarNet_L63.ipynb`: 4DVarNet implemented in Pytorch Lightning for Lorenz 63.
- `4DVarNet_L96.ipynb`: 4DVarNet implemented in Pytorch Lightning for Lorenz 96. 
-  A `gridsearch.py` Python script can be run to find optimal parameters for the CNN.

## Acknowledgements

- This project was based on [this project](https://github.com/CIA-Oceanix/DLGD2021/tree/main/data-assimilation-project).
- Many thanks to Ronan Fablet and the lecturers of his Advanced Course on Deep Learning and Geophsyical Dynamics course.


## Contact
Created by  <a href="https://github.com/simondriscoll">Simon Driscoll</a> , <a href="https://github.com/cdurand95">Charlotte Durand</a> and <a href="https://github.com/ojacquot">Oscar Jacquot</a> - feel free to contact us !

