This zip contains Split Miner 2.0 Java application, which is presented in the research article titled:

"Automated Discovery of Process Models with True Concurrency and Inclusive Choices"
A. Augusto, M. Dumas, M. La Rosa

Here follows a brief description of the files in this zip:

1. lib
	contains the libraries necessary to run the JAR applications.

2. sm2.jar
	Split Miner 2.0 java application
	
3. repair.xes.gz
	the popular public log of the repair example process, also available at http://www.promtools.org/prom6/downloads/example-logs.zip

To run Split Miner 2.0, use the following java command (in Windows):

	java -cp sm2.jar;lib\* au.edu.unimelb.services.ServiceProvider SM2 .\repair.xes.gz .\repair-model 0.05 

where:

	".\repair.xes.gz" - is the path of the input event log
	".\repair-model" - is the path of the output process model (the extension .bpmn will be added automatically by the tool)
	"0.05" - is the concurrency threshold as described in our research article mentioned above


If working on MAC, please adjust the Java command as follows:

	java -cp sm2.jar:lib/* au.edu.unimelb.services.ServiceProvider SM2 ./repair.xes.gz ./repair-model 0.05 
	
	
	
if you have questions, you can reach us at: a.augusto@unimelb.edu.au