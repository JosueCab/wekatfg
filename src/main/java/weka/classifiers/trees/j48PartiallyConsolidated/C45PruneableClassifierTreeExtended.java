package weka.classifiers.trees.j48PartiallyConsolidated;

import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.Instances;

/**
 * Class for extend handling C45PruneableClassifierTree class
 * *************************************************************************************
 *
 * @author Ander Otsoa de Alda Alzaga (ander.otsoadealda@gmail.com)
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @version $Revision: 0.3 $
 */
public class C45PruneableClassifierTreeExtended extends C45PruneableClassifierTree {

	/** for serialization */
	private static final long serialVersionUID = -4396836285687129766L;

	/** The model selection method to force the consolidated decision in a base tree */
	protected C45ModelSelectionExtended m_baseModelToForceDecision;

	/**
	 * Constructor.
	 * @param toSelectLocModel selection method for local splitting model
	 * @param pruneTree true if the tree is to be pruned
	 * @param cf the confidence factor for pruning
	 * @param raiseTree true if subtree raising has to be performed
	 * @param cleanup true if cleanup has to be done
	 * @param collapseTree true if collapse has to be done
	 * @throws Exception if something goes wrong
	 */
	public C45PruneableClassifierTreeExtended(ModelSelection toSelectLocModel, C45ModelSelectionExtended baseModelToForceDecision,
		    boolean pruneTree,float cf,
		    boolean raiseTree,
		    boolean cleanup,
            boolean collapseTree) throws Exception {
		super(toSelectLocModel, pruneTree, cf, raiseTree, cleanup, collapseTree);
		m_baseModelToForceDecision = baseModelToForceDecision;
	}

	/**
	 * Getter for m_baseModelToForceDecision member.
	 * return the model selection method to force the consolidated decision in a base tree
	 */
	public C45ModelSelectionExtended getBaseModelToForceDecision() {
		return m_baseModelToForceDecision;
	}

	/**
	 * Initializes the base tree to be build.
	 * @param data instances in the current node related to the corresponding base decision tree
	 * @param keepData  is training data to be kept?
	 */
	public void initiliazeTree(Instances data, boolean keepData) {
		if (keepData) {
			m_train = data;
		}
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
	}

	/**
	 * Setter for m_isLeaf member.
	 * @param isLeaf indicates if node is leaf
	 */
	public void setIsLeaf(boolean isLeaf) {
		m_isLeaf = isLeaf;
	}

	/**
	 * Setter for m_isEmpty member.
	 * @param isEmpty indicates if node is empty
	 */
	public void setIsEmpty(boolean isEmpty) {
		m_isEmpty = isEmpty;
	}

	/**
	 * Set m_localModel based on the consolidated model taking into account the sample.
	 * @param data instances in the current node related to the corresponding base decision tree
	 * @param consolidatedModel is the consolidated split
	 * @throws Exception if something goes wrong
	 */
	public void setLocalModel(Instances data,
			ClassifierSplitModel consolidatedModel) throws Exception {
		m_localModel = m_baseModelToForceDecision.selectModel(data, consolidatedModel);
	}

	/**
	 * Creates the vector to save the sons of the current node.
	 * @param numSons Number of sons
	 */
	public void createSonsVector(int numSons) {
		m_sons = new ClassifierTree [numSons];
	}

	/**
	 * Set given baseTree tree like the i-th son tree.
	 * @param iSon Index of the vector to save the given tree
	 * @param classifierTree the given to tree to save
	 */
	public void setIthSon(int iSon, ClassifierTree classifierTree) {
		m_sons[iSon] = classifierTree;
	}

	/**
	 * Set node as leaf
	 */
	public void setAsLeaf() {
		// Free adjacent trees
		m_sons = null;
		m_isLeaf = true;
	
		// Get NoSplit Model for tree.
		m_localModel = new NoSplit(localModel().distribution());
	}

	/**
	 * Replace current node with a given tree
	 * @param newTree the given tree to replace with
	 * @throws Exception if something goes wrong
	 */
	protected void replaceWithSubtree(C45PruneableClassifierTreeExtended newTree) throws Exception {
		m_sons = newTree.getSons();
		m_localModel = newTree.localModel();
		m_isLeaf = newTree.isLeaf();
		newDistribution(m_train);
	}

	/**
	 * Replace current node with i-th son (the largest branch)
	 * @param iSon Index of the son to replace with
	 */
	public void replaceWithIthSubtree(int iSon) throws Exception {
		replaceWithSubtree((C45PruneableClassifierTreeExtended)(son(iSon)));
	}

	/**
	 * Rebuilds the tree according to J48 algorithm and
	 *  maintaining the current tree structure
	 * @throws Exception if something goes wrong
	 */
	public void rebuildTreeFromConsolidatedStructure() throws Exception {
		if (!m_isLeaf){
			for (int iSon=0;iSon<m_sons.length;iSon++)
				((C45PruneableClassifierTreeExtended)(son(iSon))).rebuildTreeFromConsolidatedStructure();
		} else { // The current node is a leaf
			/** Build a J48 tree with the data of the current node
			 *  (based on the buildClassifier() function of the J48 class) */
			// TODO Implement the option binarySplits of J48
			// TODO Implement the option reducedErrorPruning of J48
			C45PruneableClassifierTreeExtended newTree = new C45PruneableClassifierTreeExtended(m_toSelectModel, m_baseModelToForceDecision, m_pruneTheTree, m_CF,
							    m_subtreeRaising, m_cleanup, m_collapseTheTree);
			newTree.buildClassifier(m_train);
			((C45ModelSelection)m_toSelectModel).cleanup();
			/** Replace current node with the recent built tree */
			replaceWithSubtree(newTree);
			newTree = null;
		}
	}
}