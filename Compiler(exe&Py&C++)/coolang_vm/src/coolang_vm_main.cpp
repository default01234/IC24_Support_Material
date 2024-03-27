/*  COOLang VM implementation
 Copyright (C) 2022,Han JiPeng,Beijing Huagui Technology Co., Ltd

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation, version LGPL-3.0-or-later.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <iterator>
#include <algorithm>
#include <direct.h>
#include <windows.h>
#include "coolang_vm.hpp"
#include "coolang_vm_execute.hpp"
#include <chrono>
 //#include "coolang_vm_integrate.hpp"

using namespace std;
using namespace COOLANG;
//std::map<std::set<Strg>, Strg> COOLANG::MultiTFIntegrator::knowledge_domains__dataset_filename__map;



int main(int argc, char* argv[]) {
#if debug
	std::ofstream outFile("debugoutput.txt");

	std::streambuf* coutBuf = std::cout.rdbuf();
	std::cout.rdbuf(outFile.rdbuf());

	std::streambuf* cerrBuf = std::cerr.rdbuf();
	std::cerr.rdbuf(outFile.rdbuf());
#elif filelog
	std::ofstream outFile("filecoutput.txt");

	std::streambuf* coutBuf = std::cout.rdbuf();
	std::cout.rdbuf(outFile.rdbuf());

	std::streambuf* cerrBuf = std::cerr.rdbuf();
	std::cerr.rdbuf(outFile.rdbuf());
#else
	std::ofstream outFile("output.txt");

	std::streambuf* coutBuf = std::cout.rdbuf();
	std::cout.rdbuf(outFile.rdbuf());

	std::streambuf* cerrBuf = std::cerr.rdbuf();
	std::cerr.rdbuf(outFile.rdbuf());
#endif
	cout << "argc:" << argc << endl;
	if (argc >= 2)
	{
		cout << "argv[1]:" << argv[1] << endl;
	}
	if (argc >= 3)
	{
		cout << "argv[2]:" << argv[2] << endl;
	}
	std::cout.precision(PRECISION);


	LARGE_INTEGER counter_frequency;

	LARGE_INTEGER start_grounding_process;
	LARGE_INTEGER end_grounding_process;

	LARGE_INTEGER start_execution;
	LARGE_INTEGER end_execution;

	chrono::time_point<chrono::steady_clock> start_grounding_process_ref;
	chrono::time_point<chrono::steady_clock> end_grounding_process_ref;
	chrono::time_point<chrono::steady_clock> start_execution_ref;
	chrono::time_point<chrono::steady_clock> end_execution_ref;


	QueryPerformanceFrequency(&counter_frequency);

	char coolang_exec_path_buf[2 * PATH_MAX];
	_getcwd((char*)coolang_exec_path_buf, sizeof(coolang_exec_path_buf));
	coolang_exec_path = filesystem::path(
		StringToWString(Strg(coolang_exec_path_buf)));

	HashCollection hashCollection;

	CodeTableHash& codeTableHash = hashCollection.cdth;
	ScopeTableHash& scopeTableHash = hashCollection.scpth;
	TemplateFunctionTableHash& templateFunctionTableHash = hashCollection.tfth;
	SystemTableHash& systemTableHash = hashCollection.systh;
	Setting& setting = hashCollection.setting;
	PipeCommunicationHandler* pipe_writing_handler = nullptr;
	PipeCommunicationHandler* pipe_reading_handler = nullptr;
	Executor executor(&hashCollection);
	if (argc <= 1) {
#if debug
		{
			static int ignorecount = 0;
			cerr << "main(int, char*[])" << " not enough file, exit(-1) "
				<< "\tignorecount:[" << ignorecount++ << "\t](" << __FILE__
				<< ":" << __LINE__ << ":0" << ")" << endl;
		}
#endif

		assert(false); exit(-1);

}

	filesystem::path codeFilePath(charpToPathStr(argv[1]));

	coolang_project_path = codeFilePath.parent_path();

	if (codeFilePath.extension()
		== charpToPath(COOLANG_CHARACTER_CODE_SUFFIX)) {
		hashCollection.fileKey = pathStrToStr(
			pathStr(codeFilePath.stem().c_str()));

		//if (argc >= 3) { If i don‘t comment the brackets, raise C1075, strange
		if (argc >= 3)
			setting.loadSettingFile(charpToPath(argv[2]));
		if (argc >= 3)
			setting.multiFile = false; //禁用多文件
		//}
		debugMode = setting.debugMode;

		if (setting.neuralNetworkEnabled || setting.train) {
			Strg coolang_home = getenv(COOLANG_HOME);

			pipe_writing_handler = new PipeCommunicationHandler(coolang_home,
				"order_to_neural_network_wizard.pipe", "write");
			pipe_reading_handler = new PipeCommunicationHandler(coolang_home,
				"reply_from_neural_network_wizard.pipe", "read");

			// launch neuralnetworkwizard
			/*auto cmd_ = generateFullPath(coolang_home,
				"neuralNetwork/neuralNetworkWizard.py");
			launchClientSide(cmd_, true, false, false);*/
			//manually launch the neural network wizard in developing process.
			//launch compiler (this) first, then launch neural network wizard
		}

		Scanner s(&hashCollection);
		//#if debug
		//        s.loadAssemblyCode(charpToPath("assemblyCode.coc"));
		//#else
		s.loadAssemblyCode(charpToPath(argv[1]));
		//#endif

#if debug
		{
			static int ignorecount = 0;
			std::cout << "main(int, char*[])" << "\tignorecount:["
				<< ignorecount++ << "\t](" << __FILE__ << ":" << __LINE__
				<< ":0" << ")" << " assemblyCodeTableHash:\n"
				<< s.ascth.toStrg() << std::endl;
	}
#endif
		s.scan();


		if (setting.train || setting.neuralNetworkEnabled) {

			executor.setPipeCommunicationHandler(pipe_writing_handler,
				pipe_reading_handler);

		}
		executor.setMode(MODE_GROUNDING);
		QueryPerformanceCounter(&start_grounding_process);
		start_grounding_process_ref = std::chrono::high_resolution_clock::now();
		executor.ground();
		end_grounding_process_ref = std::chrono::high_resolution_clock::now();
		QueryPerformanceCounter(&end_grounding_process);
	}
	else if (codeFilePath.extension()
		== charpToPath(COOLANG_BINARY_CODE_SUFFIX)) {
		hashCollection.loadFromBinaryFile(charpToPathStr(argv[1]));
		if (argc >= 3) {
			setting.loadSettingFile(charpToPathStr(argv[2]));
		}

	}
#if debug
	{
		static int ignorecount = 0;
		cout << "main(int, char*[])" << "\tignorecount:["
			<< ignorecount++ << "\t](" << __FILE__
			<< ":" << __LINE__ << ":0" << ")" << endl;
		codeTableHash.toStrg();
	}
#endif
	//保存编译文件
	if (setting.codeForm == 3
		&& codeFilePath.extension()
		!= charpToPath(COOLANG_BINARY_CODE_SUFFIX)) {
		if (setting.multiFile) {
			if (argc == 4 && argv[3] != nullptr) {
				hashCollection.saveToMultiFiles(StringToWString(Strg(argv[3])));
			}
			else {
				hashCollection.saveToMultiFiles(StringToWString(Strg(argv[1])));
			}
		}
		else {
			if (argc == 4 && argv[3] != nullptr) {
				hashCollection.saveToSingleFile(StringToWString(Strg(argv[3])));
			}
			else {
				hashCollection.saveToSingleFile(StringToWString(Strg(argv[1])));
			}
		}

	}
	if (argc == 2 || setting.execute == true) {
		executor.setMode(MODE_EXECUTION);
		QueryPerformanceCounter(&start_execution);
		start_execution_ref = std::chrono::high_resolution_clock::now();
		executor.execute();
		end_execution_ref = std::chrono::high_resolution_clock::now();
		QueryPerformanceCounter(&end_execution);
	}
	if (setting.exportModelingData) {
		GroundingStateSilo::flush();
	}
	//start training process before exit
	if (setting.train) {
		Strg&& train_msg_str = MessageWrapper::wrapMessage("train",
			MultiTFIntegrator::knowledge_domains__dataset_filename__map);
		pipe_writing_handler->writeMessage(train_msg_str);
	}


	if (setting.exportPerformance == true)
	{

		Numb grounding_time = 1000000 * static_cast<double>(end_grounding_process.QuadPart - start_grounding_process.QuadPart) / counter_frequency.QuadPart;
		Numb execution_time = 1000000 * static_cast<double>(end_execution.QuadPart - start_execution.QuadPart) / counter_frequency.QuadPart;
		auto grounding_time_ref = std::chrono::duration_cast<std::chrono::microseconds>(end_grounding_process_ref - start_grounding_process_ref);
		auto execution_time_ref = std::chrono::duration_cast<std::chrono::microseconds>(end_execution_ref - start_execution_ref);


		Strg performance_filename = "performance__" + toStrg(setting.CollectionCycleSerialNumber) + ".json";
		auto outputFilePath = coolang_home_path;
		outputFilePath.append(performance_filename);
		fstream fs(outputFilePath, std::ios::in | std::ios::app);
		fs.seekp(0, fs.end);

		fs << "\n{";
		fs << "\"AST_transformation_count\":" << MultiTFIntegrator::AST_transformation_count << ",\n";
		fs << "\"grounding_state_count\":" << GroundingState::grounding_state_count << ",\n";
		fs << "\"grounding_time\":" << grounding_time << ",\n";
		fs << "\"grounding_time_ref\":" << grounding_time_ref.count() << ",\n";
		fs << "\"execution_time\":" << execution_time << ",\n";
		fs << "\"execution_time_ref\":" << execution_time_ref.count() << ",\n";
		fs << "\"neural_network_assistance_count\":" << MultiTFIntegrator::neural_network_assistance_count << ",\n";
		fs << "\"ac\":" << MultiTFIntegrator::accumulated_ac / (std::max)(1, MultiTFIntegrator::neural_network_assistance_count) << ",\n";
		fs << "\"ci\":" << MultiTFIntegrator::accumulated_ci / (std::max)(1, MultiTFIntegrator::neural_network_assistance_count) << ",\n";
		fs << "\"reasoning_failure_times\":" << MultiTFIntegrator::reasoning_failure_times << ",\n";
		fs << "},";


		fs.close();
	}
	if (pipe_writing_handler) {
		pipe_writing_handler->closeConnection();
	}
	if (pipe_reading_handler) {
		pipe_reading_handler->closeConnection();
	}
	return 0;

}


