/**
 *
 */
package weka.classifiers.trees.j48ItPartiallyConsolidated;

import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedPruneableClassifierTree;
import weka.classifiers.trees.j48PartiallyConsolidated.C45ModelSelectionExtended;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PartiallyConsolidatedPruneableClassifierTree;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PruneableClassifierTreeExtended;

import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for handling a consolidated tree structure that can
 * be pruned using C4.5 procedures.
 * *************************************************************************************
 * Attention! Removed 'final' modifier from collapse() function of j48/C45PruneableClassifierTree
 *  class and from cleanup() function of j48/ClassifierTree class in order to overwrite
 *  these functions here.
 * *************************************************************************************<br/>
 *
 * @author Ander Otsoa de Alda Alzaga (ander.otsoadealda@gmail.com)
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @version $Revision: 0.3 $
 */
public class C45ItPartiallyConsolidatedPruneableClassifierTree extends
		C45PartiallyConsolidatedPruneableClassifierTree {

	/** for serialization **/
	private static final long serialVersionUID = 6410655550027990502L;

	/** Vector for storing the generated base decision trees
	 *  related to each sample */
	//protected C45PruneableClassifierTreeExtended[] m_sampleTreeVector;

	/**
	 * Constructor for pruneable consolidated tree structure. Calls
	 * the superclass constructor.
	 *
	 * @param toSelectLocModel selection method for local splitting model
	 * @param pruneTree true if the tree is to be pruned
	 * @param cf the confidence factor for pruning
	 * @param raiseTree true if subtree raising has to be performed
	 * @param cleanup true if cleanup has to be done
	 * @param collapseTree true if collapse has to be done
	 * @param numberSamples Number of Samples
	 * @throws Exception if something goes wrong
	 */
	public C45ItPartiallyConsolidatedPruneableClassifierTree(
			ModelSelection toSelectLocModel, C45ModelSelectionExtended baseModelToForceDecision,
			boolean pruneTree, float cf,
			boolean raiseTree, boolean cleanup,
			boolean collapseTree, int numberSamples) throws Exception {
		super(toSelectLocModel, baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree, numberSamples);
		// Initialize each base decision tree of the vector
		ModelSelection modelToConsolidate = ((C45ConsolidatedModelSelection)toSelectLocModel).getModelToConsolidate();
		m_sampleTreeVector = new C45PruneableClassifierTreeExtended[numberSamples];
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample] = new C45PruneableClassifierTreeExtended(
					modelToConsolidate,	baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree);
	}

	/**
	 * Builds the consolidated tree structure.
	 * (based on the method buildTree() of the class 'ClassifierTree')
	 *
	 * @param data the data for pruning the consolidated tree
	 * @param samplesVector the vector of samples used for consolidation
	 * @param keepData is training data to be kept?
	 * @throws Exception if something goes wrong
	 */
	public void buildTree(Instances data, Instances[] samplesVector, boolean keepData) throws Exception {

		ArrayList<Object[]> list = new ArrayList<>();

		// add(Data, samplesVector, tree, orderValue, currentLevel)
		list.add(new Object[] { data, samplesVector, this, null, 0 }); // The parent node is considered level 0

		//int index = 0;
		//double orderValue;

		while (list.size() > 0) {
			Object[] current = list.get(0);
			//int currentLevel = (int) current[3];
			
			/** Number of Samples. */
			Instances[] currentSamplesVector = (Instances[]) current[1];
			int numberSamples = currentSamplesVector.length;
			
			// int currentNode = (int) current[4];
			list.set(0, null); // Null to free up memory
			list.remove(0);

			Instances currentData = (Instances) current[0];
			C45ItPartiallyConsolidatedPruneableClassifierTree currentTree = (C45ItPartiallyConsolidatedPruneableClassifierTree) current[2];
			//currentTree.m_order = index;

			/** Initialize the consolidated tree */
			if (keepData) {
				currentTree.m_train = currentData;
			}
			currentTree.m_test = null;
			currentTree.m_isLeaf = false;
			currentTree.m_isEmpty = false;
			currentTree.m_sons = null;
			
			/** Initialize the base trees */
			for (int iSample = 0; iSample < numberSamples; iSample++)
				currentTree.m_sampleTreeVector[iSample].initiliazeTree(currentSamplesVector[iSample], keepData);

			/**
			 * Select the best model to split (if it is worth) based on the consolidation
			 * proccess
			 */
			currentTree.m_localModel = ((C45ConsolidatedModelSelection) currentTree.m_toSelectModel).selectModel(currentData, currentSamplesVector);
			for (int iSample = 0; iSample < numberSamples; iSample++)
				currentTree.m_sampleTreeVector[iSample].setLocalModel(currentSamplesVector[iSample], currentTree.m_localModel);

			//////////////////////////IF
			if (currentTree.m_localModel.numSubsets() > 1) {
				/** Vector storing the obtained subsamples after the split of data */
				Instances[] localInstances;
				/**
				 * Vector storing the obtained subsamples after the split of each sample of the
				 * vector
				 */
				ArrayList<Instances[]> localInstancesVector = new ArrayList<Instances[]>();

				/**
				 * For some base trees, although the current node is not a leaf, it could be
				 * empty. This is necessary in order to calculate correctly the class membership
				 * probabilities for the given test instance in each base tree
				 */
				
				ArrayList<Object[]> listSons = new ArrayList<>();

				for (int iSample = 0; iSample < numberSamples; iSample++)
					if (Utils.eq(currentTree.m_sampleTreeVector[iSample].getLocalModel().distribution().total(), 0))
						currentTree.m_sampleTreeVector[iSample].setIsEmpty(true);

				/** Split data according to the consolidated m_localModel */
				localInstances = currentTree.m_localModel.split(currentData);
				for (int iSample = 0; iSample < numberSamples; iSample++)
					localInstancesVector.add(currentTree.m_localModel.split(currentSamplesVector[iSample]));

				/**
				 * Create the child nodes of the current node and call recursively to
				 * getNewTree()
				 */
				currentData = null;
				currentSamplesVector = null;
				currentTree.m_sons = new ClassifierTree[currentTree.m_localModel.numSubsets()];
				for (int iSample = 0; iSample < numberSamples; iSample++)
					((C45PruneableClassifierTreeExtended) currentTree.m_sampleTreeVector[iSample])
							.createSonsVector(currentTree.m_localModel.numSubsets());
				
				
				//////////////////
				for (int iSon = 0; iSon < currentTree.m_sons.length; iSon++) {
					/** Vector storing the subsamples related to the iSon-th son */
					Instances[] localSamplesVector = new Instances[numberSamples];
					for (int iSample = 0; iSample < numberSamples; iSample++)
						localSamplesVector[iSample] = ((Instances[]) localInstancesVector.get(iSample))[iSon];
					
					//getNewTree
				

					C45ModelSelectionExtended baseModelToForceDecision = currentTree.m_sampleTreeVector[0].getBaseModelToForceDecision();
					C45ItPartiallyConsolidatedPruneableClassifierTree newTree = new C45ItPartiallyConsolidatedPruneableClassifierTree(
							currentTree.m_toSelectModel, baseModelToForceDecision, m_pruneTheTree, m_CF, m_subtreeRaising, m_cleanup,
							m_collapseTheTree, localSamplesVector.length);
					/** Set the recent created base trees like the sons of the given parent node */
					for (int iSample = 0; iSample < numberSamples; iSample++)
						((C45PruneableClassifierTreeExtended) currentTree.m_sampleTreeVector[iSample]).setIthSon(iSon,
								newTree.m_sampleTreeVector[iSample]);
					
					listSons.add(new Object[] { localInstances[iSon], localSamplesVector, newTree, 0, 0 });
					
					currentTree.m_sons[iSon] = newTree;

					///////////////////
					
					localInstances[iSon] = null;
					localSamplesVector = null;
				}
				
				listSons.addAll(list);
				list = listSons;
				localInstances = null;
				localInstancesVector.clear();
				listSons = null;

			} else {
				currentTree.m_isLeaf = true;
				for (int iSample = 0; iSample < numberSamples; iSample++)
					currentTree.m_sampleTreeVector[iSample].setIsLeaf(true);

				if (Utils.eq(currentTree.m_localModel.distribution().total(), 0)) {
					currentTree.m_isEmpty = true;
					for (int iSample = 0; iSample < numberSamples; iSample++)
						currentTree.m_sampleTreeVector[iSample].setIsEmpty(true);
				}
				currentData = null;
				currentSamplesVector = null;
			}
			//index++;

		}
	System.out.println("END buildTree");	
	}


}