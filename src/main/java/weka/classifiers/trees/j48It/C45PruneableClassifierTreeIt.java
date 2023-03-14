package weka.classifiers.trees.j48It;

import java.util.ArrayList;
import java.util.Stack;

import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for handling a consolidated tree structure that can
 * be pruned using C4.5 procedures.
 * *************************************************************************************
 * 
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus) 
 * @version $Revision: 1.2 $
 */
public class C45PruneableClassifierTreeIt extends
		C45PruneableClassifierTree {

	/** for serialization */
	private static final long serialVersionUID = 2660972525647728377L;

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
	 * @throws Exception if something goes wrong
	 */
	public C45PruneableClassifierTreeIt(
			ModelSelection toSelectLocModel, boolean pruneTree, float cf,
			boolean raiseTree, boolean cleanup, boolean collapseTree) throws Exception {
		super(toSelectLocModel, pruneTree, cf, raiseTree, cleanup, collapseTree);
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
	 public void buildTree(Instances data, boolean keepData) throws Exception {
	 	    Stack<Object[]> stack = new Stack<>();
	 	    stack.push(new Object[] {data, this});

	 	    while (!stack.isEmpty()) {
	 	        Object[] current = stack.pop();
	 	        Instances currentData = (Instances) current[0];
	 	        C45PruneableClassifierTreeIt currentTree = (C45PruneableClassifierTreeIt) current[1];

	 	        Instances[] localInstances;
	 	        if (keepData) {
	 	            currentTree.m_train = currentData;
	 	        }
	 	        currentTree.m_test = null;
	 	        currentTree.m_isLeaf = false;
	 	        currentTree.m_isEmpty = false;
	 	        currentTree.m_sons = null;
	 	        currentTree.m_localModel = currentTree.m_toSelectModel.selectModel(currentData);
	 	        if (currentTree.m_localModel.numSubsets() > 1) {
	 	            localInstances = currentTree.m_localModel.split(currentData);
	 	            currentData = null;
	 	            currentTree.m_sons = new ClassifierTree[currentTree.m_localModel.numSubsets()];
	 	            for (int i = 0; i < currentTree.m_sons.length; i++) {
	 	                ClassifierTree newTree = new C45PruneableClassifierTreeIt(currentTree.m_toSelectModel, m_pruneTheTree, m_CF,
	 	  				     m_subtreeRaising, m_cleanup, m_collapseTheTree);
	 	                stack.push(new Object[] {localInstances[i], newTree});
	 	                currentTree.m_sons[i] = newTree;
	 	            }
	 	        } else {
	 	            currentTree.m_isLeaf = true;
	 	            if (Utils.eq(currentData.sumOfWeights(), 0)) {
	 	                currentTree.m_isEmpty = true;
	 	            }
	 	            currentData = null;
	 	        }
	 	    }
	 	}

}
