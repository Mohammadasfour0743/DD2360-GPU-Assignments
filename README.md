# Assignment 1

## Question 1

Open the provided notebook in Google Colab.

Select a GPU runtime (T4) from the Colab settings (Runtime -> Change runtime type).

Run the cell containing the %%writefile vectorAdd.cu command to generate the source file.

Compile the CUDA code by running the following command in a new cell: !nvcc vectorAdd.cu -o vectorAdd.

Execute the compiled program to view the execution time breakdown across different vector lengths: !./vectorAdd.

## Question 2

1. Copy/Open the notebook in Google Collab
2. Select the T4 execution type from Collab settings
3. Run cells in order while commenting out or adjusting the matrix sizes in `test_sizes[][3]`
