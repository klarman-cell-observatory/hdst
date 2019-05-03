# ST Aligner

This package can be used to find approximate coordinates of spots on an HDST array.

## Installation

To install the package and its dependencies with pip, run

```
pip install <path-to-staligner-repository>
```

## Usage

ST Aligner is run on the bright-field microscopy image from an HDST experiment.
Before proceeding, make sure that the microscopy image has the right orientation; spots will be indexed from the top left in the output file.

Invoke the alignment script by running

```
staligner --input <path-to-bright-field-image> --output <output-directory> --annotate
```

The `--annotate` flag is optional but recommended.
When specified, ST Aligner will emit an annotated bright-field image, showing the inferred locations of the spots.
The annotated image can be used to verify that the results are correct.
