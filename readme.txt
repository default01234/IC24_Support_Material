Hello! If you're interested in reproducing the experiments from our paper, please follow the steps outlined below:

1. Find a suitable location on your windows system and copy the COOLANG_HOME directory there.

2. Edit your environment variables to include COOLANG_HOME. This directory acts as a reference point for the compiler to locate its necessary components.

3. If you have COOLang code you wish to compile, use the following command:
   `compiler_file_combination.exe "path_to_your_code.cos" "path_to_your_settings_file.setting"`

4. The "test" directory includes the COOLang code required for this experiment.

5. Compile the "coolang_vm" as a Visual Studio project. Ensure that you have the required libraries.

6. When you have a compiled COOLang file (with a `.coc` extension), you can execute it using:
   `coolang_vm.exe "path_to_your_coc_file.coc" "path_to_your_settings_file.setting "`

7. If your settings file has training or neural network enabled, the system will create a pipeline waiting for a Python process to connect.

8. Execute the Python script for neural network training using:
   `GUIDEDGROUNDING/neuralNetworkWizard.py "path_to_your_settings_file.setting"`

Please note that the neural network mode should not be directly enabled without first using the training mode to construct the DSNN with the provided Python files.

We've included our experimental results for comparison, and you should expect to see similar outputs from your own runs.

If you're interested in exploring COOL further, an older version is available. You can find the MSI installer included with this package. However, this version has many known issues and lacks AI support. We are grateful for your interest in COOL, and we anticipate releasing a new MSI version with AI features in the future.

THANK YOU
