# SCAMPy

| **Build Status**                    |
|:------------------------------------|
| [![travis][travis-img]][travis-url] |

[travis-img]: https://travis-ci.org/charleskawczynski/SCAMPy.svg?branch=master
[travis-url]: https://travis-ci.org/charleskawczynski/SCAMPy

## Description

SCAMPy (Single Column Atmospheric Model in Python) provides a framework for testing parameterizations of clouds and turbulence.
It is particularly designed to support eddy-diffusivity mass-flux modeling frameworks.

Information about the EDMF parameterization implemented in SCAMPy can be found in:

Tan, Z., C. M. Kaul, K. G. Pressel, Y. Cohen, T. Schneider, and J. Teixeira, 2018: An extended eddy-diffusivity mass-flux scheme for unified representation of subgrid-scale turbulence and convection. Journal of Advances in Modeling Earth Systems, in press.

The code is written in _Python_.

## Install

`git clone https://github.com/charleskawczynski/SCAMPy`

## Running

Choose one of:

 - `python main.py Soares`
 - `python main.py Bomex`
 - `python main.py life_cycle_Tan2018`
 - `python main.py Rico`
 - `python main.py TRMM_LBA`
 - `python main.py ARM_SGP`
 - `python main.py GATE_III`
 - `python main.py DYCOMS_RF01`
 - `python main.py GABLS`
 - `python main.py SP`

## Testing

To make sure that local changes haven't broken the build, use

`pytest`

To update tests, update

 - code configuration for a particular `case` and
 - `sol_expected[case]` in `test_results.py`

Cases currently tested:

|  **Verified** | **Tested** | **Case name**        |
|:--------------|------------|----------------------|
|      [ ]      |     [x]    | `Soares`             |
|      [x]      |     [x]    | `Bomex`              |
|      [ ]      |     [x]    | `life_cycle_Tan2018` |
|      [ ]      |     [x]    | `Rico`               |
|      [ ]      |     [x]    | `TRMM_LBA`           |
|      [ ]      |     [ ]    | `ARM_SGP`            |
|      [ ]      |     [ ]    | `GATE_III`           |
|      [ ]      |     [ ]    | `DYCOMS_RF01`        |
|      [ ]      |     [ ]    | `GABLS`              |
|      [ ]      |     [ ]    | `SP`                 |

## Acknowledgments

Code Contributors:
	Colleen Kaul (Caltech)--initial/primary developer. Inquiries may be sent to cmkaul@caltech.edu;
	Yair Cohen (Caltech);
	Anna Jaruga (JPL/Caltech);
	Kyle Pressel (Caltech);
	Zhihong Tan (U. Chicago)

Additional Acknowledgements:
	Tapio Schneider (Caltech);
	Joao Teixeira (JPL)

