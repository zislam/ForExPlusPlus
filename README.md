# ForExPlusPlus

Implementation of the knowledge discovery framework ForEx++, which was published in:

*Md Nasim Adnan and Md Zahidul Islam: ForEx++: A New Framework for Knowledge Discovery from Decision Forests In: Australasian Journal of Information Systems Vol 21, 2017.*

This algorithm processes a decision forest and provides a list of high-quality rules that account for each class.

## BibTeX
```
@article{adnan2017forex,
  title={ForEx++: A New Framework for Knowledge Discovery from Decision Forests},
  author={Adnan, Md Nasim and Islam, Md Zahidul},
  journal={Australasian Journal of Information Systems (AJIS)},
  volume={21},
  pages={1--20},
  year={2017}
}
```

## Installation 

Either download ForExPlusPlus from the Weka package manager, or download the latest release from the "Releases" section on the sidebar of Github. A video on the installation and use of the package can be found [here](https://www.youtube.com/watch?v=kHIkpZuLPMQ&t=0s).

## Compilation / Development

This repository contains a Netbeans project. Import into Netbeans and include weka.jar, SysFor.jar, and ForestPA.jar as compile-time libraries. SysFor.jar and ForestPA.jar are available in the Weka package manager.

## Valid options are:

`-P` 
Whether to print the decision forest that the ForEx++ rules were selected from
 (default false)

`-Z`
 Whether to remove rules with no coverage before calculating mean coverage, 
 support, and rule length
 (default true)

`-GC`
 Whether to group rules by class value in the final output.
 (default true)

`-E <acc | cov | len>`
 Sort Method for Displaying Rules.
 (Default = sort by rule accuracy)

`-UA`
 Whether to use accuracy in selecting ForEx++ rules
 (default true)

`-UC`
 Whether to use coverage in selecting ForEx++ rules
 (default true)

`-UR`
 Whether to use rule length in selecting ForEx++ rules
 (default true)
