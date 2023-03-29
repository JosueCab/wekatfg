/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    J48Consolidated.java
 *    Copyright (C) 2021 ALDAPA Team (http://www.aldapa.eus)
 *    Faculty of Informatics, Donostia, 20018
 *    University of the Basque Country (UPV/EHU), Basque Country
 *    
 */

package weka.classifiers.trees;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.BinC45ModelSelection;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedPruneableClassifierTree;
import weka.classifiers.trees.j48Consolidated.InstancesConsolidated;
import weka.classifiers.trees.j48It.C45PruneableClassifierTreeIt;
import weka.core.AdditionalMeasureProducer;
import weka.core.Drawable;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Summarizable;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.gui.ProgrammaticProperty;

/**
<!-- globalinfo-start -->
 * Class for generating a pruned or unpruned C4.5 tree iteratively. 
<!-- globalinfo-end -->
*/

public class J48It
extends J48
implements OptionHandler, Drawable, Matchable, Sourcable, 
WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, 
TechnicalInformationHandler {


	/**
	   * Generates the classifier.
	   * 
	   * @param instances the data to train the classifier with
	   * @throws Exception if classifier can't be built successfully
	   */
	  @Override
	  public void buildClassifier(Instances instances) throws Exception {

	    if ((m_unpruned) && (!m_subtreeRaising)) {
	      throw new Exception("Subtree raising does not need to be unset for unpruned trees!");
	    }
	    if ((m_unpruned) && (m_reducedErrorPruning)) {
	      throw new Exception("Unpruned tree and reduced error pruning cannot be selected simultaneously!");
	    }
	    if ((m_unpruned) && (m_CF != 0.25f)) {
	      throw new Exception("It does not make sense to change the confidence for an unpruned tree!");
	    }
	    if ((m_reducedErrorPruning) && (m_CF != 0.25f)) {
	      throw new Exception("Changing the confidence does not make sense for reduced error pruning.");
	    }
	    if ((!m_reducedErrorPruning) && (m_numFolds != 3)) {
	      throw new Exception("Changing the number of folds does not make sense if"
	              + " reduced error pruning is not selected.");
	    }
	    if ((!m_reducedErrorPruning) && (m_Seed != 1)) {
	      throw new Exception("Changing the seed does not make sense if"
	              + " reduced error pruning is not selected.");
	    }
	    if ((m_CF <= 0) || (m_CF >= 1)) {
	      throw new Exception("Confidence has to be greater than zero and smaller than one!");
	    }
	    getCapabilities().testWithFail(instances);

	    ModelSelection modSelection;

	    if (m_binarySplits) {
	      modSelection = new BinC45ModelSelection(m_minNumObj, instances,
	        m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
	    } else {
	      modSelection = new C45ModelSelection(m_minNumObj, instances,
	        m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
	    }
	    if (!m_reducedErrorPruning) {
	      m_root = new C45PruneableClassifierTreeIt(modSelection, !m_unpruned, m_CF,
	        m_subtreeRaising, !m_noCleanup, m_collapseTree);
	    } else {
	      m_root = new PruneableClassifierTree(modSelection, !m_unpruned,
	        m_numFolds, !m_noCleanup, m_Seed);
	    }
	    m_root.buildClassifier(instances);
	    if (m_binarySplits) {
	      ((BinC45ModelSelection) modSelection).cleanup();
	    } else {
	      ((C45ModelSelection) modSelection).cleanup();
	    }
	  }

}