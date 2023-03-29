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
 * Class for generating a pruned or unpruned C4.5 consolidated tree. Uses the Consolidated Tree Construction (CTC) algorithm: a single tree is built based on a set of subsamples. New options are added to the J48 class to set the Resampling Method (RM) for the generation of samples to be used in the consolidation process. For more information, see:<br/>
 * <br/>
 * Jes&uacute;s M. P&eacute;rez and Javier Muguerza and Olatz Arbelaitz and Ibai Gurrutxaga and Jos&eacute; I. Mart&iacute;­n.  
 * "Combining multiple class distribution modified subsamples in a single tree". Pattern Recognition Letters (2007), 28(4), pp 414-422.
 * <a href="http://dx.doi.org/10.1016/j.patrec.2006.08.013" target="_blank">doi:10.1016/j.patrec.2006.08.013</a>
 * <p/>
 * A new way has been added to determine the number of samples to be used in the consolidation process which guarantees the minimum percentage, the coverage value, of the examples of the original sample to be contained by the set of built subsamples. For more information, see:<br/>
 * <br/>
 * Igor Ibarguren and Jes&uacute;s M. P&eacute;rez and Javier Muguerza and Ibai Gurrutxaga and Olatz Arbelaitz.  
 * "Coverage-based resampling: Building robust consolidated decision trees". Knowledge Based Systems (2015), Vol. 79, pp 51-67.
 * <a href="http://dx.doi.org/10.1016/j.knosys.2014.12.023" target="_blank">doi:10.1016/j.knosys.2014.12.023</a>
 * <p/>
<!-- globalinfo-end -->
 *
<!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Perez2007,
 *    title = "Combining multiple class distribution modified subsamples in a single tree",
 *    journal = "Pattern Recognition Letters",
 *    volume = "28",
 *    number = "4",
 *    pages = "414 - 422",
 *    year = "2007",
 *    doi = "10.1016/j.patrec.2006.08.013",
 *    author = "Jes\'us M. P\'erez and Javier Muguerza and Olatz Arbelaitz and Ibai Gurrutxaga and Jos\'e I. Mart\'i­n"
 * }
 * </pre>
 * <p/>
 * <pre>
 * &#64;article{Ibarguren2015,
 *    title = "Coverage-based resampling: Building robust consolidated decision trees",
 *    journal = "Knowledge Based Systems",
 *    volume = "79",
 *    pages = "51 - 67",
 *    year = "2015",
 *    doi = "10.1016/j.knosys.2014.12.023",
 *    author = "Igor Ibarguren and Jes\'us M. P\'erez and Javier Muguerza and Ibai Gurrutxaga and Olatz Arbelaitz"
 * }
 * </pre>
 * <p/>
<!-- technical-bibtex-end -->
 * *************************************************************************************<br/>
<!-- options-start -->
 * Valid options are: <p/>
 * 
 * J48 options <br/>
 * ==========
 *
 * <pre>
 * -U
 *  Use unpruned tree.
 * </pre>
 * 
 * <pre>
 * -O
 *  Do not collapse tree.
 * </pre>
 * 
 * <pre>
 * -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)
 * </pre>
 * 
 * <pre>
 * -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)
 *  </pre>
 *  
 * <pre>
 * -S
 *  Don't perform subtree raising.
 * </pre>
 * 
 * <pre>
 * -L
 *  Do not clean up after the tree has been built.
 * </pre>
 * 
 * <pre>
 * -J
 *  Do not use MDL correction for info gain on numeric attributes.
 * </pre>
 * 
 * <pre>
 * -A
 *  Laplace smoothing for predicted probabilities.
 * </pre>
 * 
 * <pre>
 * -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).
 * </pre>
 * 
 * <pre>
 * -doNotMakeSplitPointActualValue
 *  Do not make split point actual value.
 * </pre>
 * 
 * Options to set the Resampling Method (RM) for the generation of samples
 *  to use in the consolidation process <br/>
 * =============================================================================================== 
 * <pre> -RM-C
 *  Determines the way to set the number of samples to be generated will be based on
 *  a coverage value as a percentage. In the case this option is not set, the number of samples
 *  will be determined using a fixed value. 
 *  (set by default)</pre>
 * 
 * <pre> -RM-N &lt;number of samples&gt;
 *  Number of samples to be generated for the use in the construction of the consolidated tree.
 *  It can be set as a fixed value or based on a coverage value as a percentage, when -RM-C option
 *  is used, which guarantees the number of samples necessary to adequately cover the examples 
 *  of the original sample
 *  (default 5 for a fixed value or 99% for the case based on a coverage value)</pre>
 * 
 * <pre> -RM-R
 *  Determines whether or not replacement is used when generating the samples.
 *  (default false)</pre>
 * 
 * <pre> -RM-B &lt;Size of each sample(&#37;)&gt;
 *  Size of each sample(bag), as a percentage of the training set size.
 *  Combined with the option &lt;distribution minority class&gt; accepts:
 *  * -1 (sizeOfMinClass): The size of the minority class  
 *  * -2 (Max): Maximum size taking &lt;distribution minority class&gt; into account
 *  *           and using no replacement
 *  (default -2)</pre>
 *  
 * <pre> -RM-D &lt;distribution minority class&gt;
 *  Determines the new value of the distribution of the minority class, if we want to change it.
 *  It can be one of the following values:
 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples
 *    (this option can only be used with binary problems (two-class datasets))
 *  * -1 (free): Works with the instances without taking their class into account
 *  * -2 (stratified): Maintains the original class distribution in the new samples
 *  (default 50.0) 
 * 
<!-- options-end -->
 *
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus)
 * @author Igor Ibarguren (igor.ibarguren@ehu.eus) 
 *  (based on the previous version written in colaboration with Fernando Lozano)
 *  (based on J48.java written by Eibe Frank)
 * @version $Revision: 3.2 $
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