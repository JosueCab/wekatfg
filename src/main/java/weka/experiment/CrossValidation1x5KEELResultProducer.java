/**
 * 
 */
package weka.experiment;

import java.io.File;

import weka.core.AdditionalMeasureProducer;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.RevisionHandler;
import weka.core.Utils;
import weka.core.WekaException;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * @author txus
 *
 */
public class CrossValidation1x5KEELResultProducer extends CrossValidationResultProducer implements ResultProducer,
OptionHandler, AdditionalMeasureProducer, RevisionHandler {

	/** for serialization */
	private static final long serialVersionUID = -8011665562910923690L;


	/** File related to the dataset of interest, in fact, the training sample of fold 1 (out of 5).
	 * 	Necessary to obtain the rest of the training and test samples.
	 */
	protected File m_fDataset;

	/**
	 * Sets file related to the dataset of interest.
 	 * 
	 * @param f file of dataset.
	 */
	public void setDataset(File f) {

		m_fDataset = f;
	}

	/**
	 * Gets the results for a specified run number. Different run numbers
	 * correspond to different train and test samples of the data. Results produced should
	 * be sent to the current ResultListener
	 * 
	 * @param run the run number to get results for.
	 * @throws Exception if a problem occurs while getting the results
	 */
	@Override
	public void doRun(int run) throws Exception {

		if (getRawOutput()) {
			if (m_ZipDest == null) {
				m_ZipDest = new OutputZipper(m_OutputFile);
			}
		}

		if (m_Instances == null) {
			throw new Exception("No Instances set");
		}
		
		if (!m_Instances.classAttribute().isNominal()) {
			System.err.println("File '" + m_fDataset.getName() + "' doesn't contain a classification problem!");
		}
	    String path = (new File(m_fDataset.getParent())).getPath() + File.separator;
	    String datasetDir = (new File(m_fDataset.getParent())).getName();
	    String contextDir = (new File(m_fDataset.getParent())).getParent();
		// Index of context: 1.standard || 2.imbalanced || 3.imbalanced-preprocessed
		int i_context = 0;
		if (contextDir.indexOf("standard") >= 0)
			i_context = 1;
		else {
			if ((contextDir.indexOf("imbalanced") >= 0) &&
				(contextDir.indexOf("preprocessed") < 0) &&
				(contextDir.indexOf("SMOTE") < 0))
				i_context = 2;
			else {
				if ((contextDir.indexOf("imbalanced") >= 0) &&
						((contextDir.indexOf("preprocessed") >= 0) ||
						(contextDir.indexOf("SMOTE") >= 0)))
						i_context = 3;
				else
					System.err.println("File '" + m_fDataset + "' doesn't belong to any known context!");
			}
		}
		// The filename's pattern is different for '3.imbalanced-preprocessed'context
	    String pattern = (i_context == 3) ? "0s0.tra.dat" : "1tra.dat";
		int index = m_fDataset.getName().lastIndexOf(pattern);
		String basename = m_fDataset.getName().substring(0, index);
		
		for (int fold = 0; fold < m_NumFolds; fold++) {
			// Add in some fields to the key like run and fold number, dataset name
			Object[] seKey = m_SplitEvaluator.getKey();
			Object[] key = new Object[seKey.length + 3];
			//key[0] = Utils.backQuoteChars(m_Instances.relationName()); // (Almost) always 'unknow'
			// In '3.imbalanced-preprocessed' context, for all databases the names of the samples are the same.
			key[0] = (i_context == 3) ? i_context + "-" + datasetDir : i_context + "-" + m_fDataset.getName();
			key[1] = "" + run;
			key[2] = "" + (fold + 1);
			System.arraycopy(seKey, 0, key, 3, seKey.length);
			if (m_ResultListener.isResultRequired(this, key)) {
				String filename;
				File file;
				Instances train;
				Instances test;
				// training set
				if (fold == 0)
					train = m_Instances;
				else {
					// The filename's pattern is different for '3.imbalanced-preprocessed' context
					filename = (i_context == 3) ? path + basename + fold + "s0.tra.dat" : path + basename + (fold + 1) + "tra.dat";
					file =  new File(filename);
					if (!file.exists()) {
						throw new WekaException("Training set '" + filename + "' not found!");
					}
					train = DataSource.read(filename);
				}
				// test set
				// The filename's pattern is different for '3.imbalanced-preprocessed' context
				filename = (i_context == 3) ? path + basename + fold + "s0.tst.dat" : path + basename + (fold + 1) + "tst.dat";
				file =  new File(filename);
				if (!file.exists()) {
					throw new WekaException("Test set '" + filename + "' not found!");
				}
				test = DataSource.read(filename);
				// test headers
				if (!train.equalHeaders(test)) {
					throw new WekaException("Train and test set (= " + filename + ") "
							+ "are not compatible:\n" + train.equalHeadersMsg(test));
				}
				try {
					Object[] seResults = m_SplitEvaluator.getResult(train, test);
					Object[] results = new Object[seResults.length + 1];
					results[0] = getTimestamp();
					System.arraycopy(seResults, 0, results, 1, seResults.length);
					if (m_debugOutput) {
						//String resultName = ("" + run + "." + (fold + 1) + "."
						//		+ Utils.backQuoteChars(m_Instances.relationName())+ "." + m_SplitEvaluator
						//		.toString()).replace(' ', '_');
						// m_Instances.relationName() (Almost) always 'unknow' in 1x5CV tra/tst KEEL samples
						// In '3.imbalanced-preprocessed' context, for all databases the names of the samples are the same.
						String resultName;
						if (i_context == 3)
							resultName = ("" + run + "." + (fold + 1) + "."
									+ i_context + "-" + datasetDir + "." + m_SplitEvaluator
											.toString()).replace(' ', '_');
						else
							resultName = ("" + run + "." + (fold + 1) + "."
									+ i_context + "-" + m_fDataset.getName() + "." + m_SplitEvaluator
											.toString()).replace(' ', '_');
						resultName = Utils.removeSubstring(resultName, "weka.classifiers.");
						resultName = Utils.removeSubstring(resultName, "weka.filters.");
						resultName = Utils.removeSubstring(resultName,
								"weka.attributeSelection.");
						m_ZipDest.zipit(m_SplitEvaluator.getRawResultOutput(), resultName);
					}
					m_ResultListener.acceptResult(this, key, results);
				} catch (Exception ex) {
					// Save the train and test datasets for debugging purposes?
					throw ex;
				}
			}
		}
	}

}
