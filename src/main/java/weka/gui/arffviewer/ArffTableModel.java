/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * ArffTableModel.java
 * Copyright (C) 2005-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.arffviewer;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Undoable;
import weka.core.Utils;
import weka.core.converters.AbstractFileLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.gui.ComponentHelper;

import javax.swing.*;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.DefaultTableModel;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Vector;

/**
 * The model for the Arff-Viewer.
 *
 * 
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 15572 $
 */
public class ArffTableModel extends DefaultTableModel implements Undoable {

  /** for serialization. */
  private static final long serialVersionUID = 3411795562305994946L;

  /** the listeners */
  protected HashSet<TableModelListener> m_Listeners;

  /** the data */
  protected Instances m_Data;

  /** whether notfication is enabled */
  protected boolean m_NotificationEnabled;

  /** whether undo is active */
  protected boolean m_UndoEnabled;

  /** whether to ignore changes, i.e. not adding to undo history */
  protected boolean m_IgnoreChanges;

  /** the undo list (contains temp. filenames) */
  protected Vector<File> m_UndoList;

  /** whether the table is read-only */
  protected boolean m_ReadOnly;

  /** whether to display the attribute index in the table header. */
  protected boolean m_ShowAttributeIndex;

  /** whether to show attribute weights. */
  protected boolean m_ShowAttributeWeights;

  /** whether to show instance weights. */
  protected boolean m_ShowInstanceWeights;

  /**
   * for caching long relational and string values that get processed for
   * display.
   */
  protected Hashtable<String, String> m_Cache;

  /**
   * performs some initialization
   */
  private ArffTableModel() {
    super();

    m_Listeners = new HashSet<TableModelListener>();
    m_Data = null;
    m_NotificationEnabled = true;
    m_UndoList = new Vector<File>();
    m_IgnoreChanges = false;
    m_UndoEnabled = true;
    m_ReadOnly = false;
    m_ShowAttributeIndex = false;
    m_ShowAttributeWeights = false;
    m_ShowInstanceWeights = false;
    m_Cache = new Hashtable<String, String>();
  }

  /**
   * initializes the object and loads the given file
   * 
   * @param filename the file to load
   * @param loaders optional varargs for a loader to use
   */
  public ArffTableModel(String filename, AbstractFileLoader... loaders) {
    this();

    if ((filename != null) && (!filename.equals(""))) {
      loadFile(filename, loaders);
    }
  }

  /**
   * initializes the model with the given data
   * 
   * @param data the data to use
   */
  public ArffTableModel(Instances data) {
    this();

    this.m_Data = data;
    for (int i = 0; i < data.numAttributes(); i++) {
      if (data.attribute(i).weight() != 1.0) {
        m_ShowAttributeWeights = true;
        break;
      }
    }
    for (int i = 0; i < data.numInstances(); i++) {
      if (data.instance(i).weight() != 1.0) {
        m_ShowInstanceWeights = true;
        break;
      }
    }
  }

  /**
   * returns whether the notification of changes is enabled
   * 
   * @return true if notification of changes is enabled
   */
  public boolean isNotificationEnabled() {
    return m_NotificationEnabled;
  }

  /**
   * sets whether the notification of changes is enabled
   * 
   * @param enabled enables/disables the notification
   */
  public void setNotificationEnabled(boolean enabled) {
    m_NotificationEnabled = enabled;
  }

  /**
   * returns whether undo support is enabled
   * 
   * @return true if undo support is enabled
   */
  @Override
  public boolean isUndoEnabled() {
    return m_UndoEnabled;
  }

  /**
   * sets whether undo support is enabled
   * 
   * @param enabled whether to enable/disable undo support
   */
  @Override
  public void setUndoEnabled(boolean enabled) {
    m_UndoEnabled = enabled;
  }

  /**
   * returns whether the model is read-only
   * 
   * @return true if model is read-only
   */
  public boolean isReadOnly() {
    return m_ReadOnly;
  }

  /**
   * sets whether the model is read-only
   * 
   * @param value if true the model is set to read-only
   */
  public void setReadOnly(boolean value) {
    m_ReadOnly = value;
  }

  /**
   * loads the specified ARFF file
   * 
   * @param filename the file to load
   * @param loaders optional varargs for a loader to use
   */
  protected void loadFile(String filename, AbstractFileLoader... loaders) {
    AbstractFileLoader loader;

    if (loaders == null || loaders.length == 0) {
      loader = ConverterUtils.getLoaderForFile(filename);
    } else {
      loader = loaders[0];
    }

    if (loader != null) {
      try {
        loader.setFile(new File(filename));
        setInstances(loader.getDataSet());
      } catch (Exception e) {
        ComponentHelper.showInformationBox(null, "Error loading file...", e.toString(),
            JOptionPane.ERROR_MESSAGE);
        System.out.println(e);
        setInstances(null);
      }
    }
  }

  /**
   * sets the data
   * 
   * @param data the data to use
   */
  public void setInstances(Instances data) {
    m_Data = data;
    m_ShowAttributeWeights = false;
    for (int i = 0; i < data.numAttributes(); i++) {
      if (data.attribute(i).weight() != 1.0) {
        m_ShowAttributeWeights = true;
        break;
      }
    }
    m_ShowInstanceWeights = false;
    for (int i = 0; i < data.numInstances(); i++) {
      if (data.instance(i).weight() != 1.0) {
        m_ShowInstanceWeights = true;
        break;
      }
    }
    m_Cache.clear();
    fireTableDataChanged();
  }

  /**
   * returns the data
   * 
   * @return the current data
   */
  public Instances getInstances() {
    return m_Data;
  }

  /**
   * Returns the attribute index for the given column index.
   *
   * @param columnIndex the column index
   *
   * @return the attribute index
   */
  public int getAttributeIndex(int columnIndex) {

    if (m_ShowInstanceWeights) {
      return columnIndex - 2;
    } else {
      return columnIndex - 1;
    }
  }

  /**
   * Check if given index is in range of column indices for attributes
   *
   * @param columnIndex the column index
   *
   * @return true if the column corresponds to attribute
   */
  public boolean isAttribute(int columnIndex) {

    if (m_ShowInstanceWeights) {
      return (columnIndex > 1) && (columnIndex < getColumnCount());
    } else {
      return (columnIndex > 0) && (columnIndex < getColumnCount());
    }
  }

  /**
   * returns the attribute at the given index, can be NULL if not an attribute
   * column
   * 
   * @param columnIndex the index of the column
   * @return the attribute at the position
   */
  public Attribute getAttributeAt(int columnIndex) {
    if (isAttribute(columnIndex)) {
      return m_Data.attribute(getAttributeIndex(columnIndex));
    } else {
      return null;
    }
  }

  /**
   * returns the TYPE of the attribute at the given position
   * 
   * @param columnIndex the index of the column
   * @return the attribute type
   */
  public int getType(int columnIndex) {
    return getType(-1, columnIndex);
  }

  /**
   * returns the TYPE of the attribute at the given position
   * 
   * @param rowIndex the index of the row
   * @param columnIndex the index of the column
   * @return the attribute type
   */
  public int getType(int rowIndex, int columnIndex) {
    int result;

    result = Attribute.STRING;

    if ((rowIndex < 0) && isAttribute(columnIndex)) {
      result = m_Data.attribute(getAttributeIndex(columnIndex)).type();
    } else if ((rowIndex >= 0) && (rowIndex < getRowCount()) && isAttribute(columnIndex)) {
      result = m_Data.instance(rowIndex).attribute(getAttributeIndex(columnIndex)).type();
    }

    return result;
  }

  /**
   * deletes the attribute at the given col index. notifies the listeners.
   * 
   * @param columnIndex the index of the attribute to delete
   */
  public void deleteAttributeAt(int columnIndex) {
    deleteAttributeAt(columnIndex, true);
  }

  /**
   * deletes the attribute at the given col index
   * 
   * @param columnIndex the index of the attribute to delete
   * @param notify whether to notify the listeners
   */
  public void deleteAttributeAt(int columnIndex, boolean notify) {
    if (isAttribute(columnIndex)) {
      if (!m_IgnoreChanges) {
        addUndoPoint();
      }
      m_Data.deleteAttributeAt(getAttributeIndex(columnIndex));
      if (notify) {
        notifyListener(new TableModelEvent(this, TableModelEvent.HEADER_ROW));
      }
    }
  }

  /**
   * deletes the attributes at the given indices
   * 
   * @param columnIndices the column indices
   */
  public void deleteAttributes(int[] columnIndices) {
    int i;

    Arrays.sort(columnIndices);

    addUndoPoint();

    m_IgnoreChanges = true;
    for (i = columnIndices.length - 1; i >= 0; i--) {
      deleteAttributeAt(columnIndices[i], false);
    }
    m_IgnoreChanges = false;

    notifyListener(new TableModelEvent(this, TableModelEvent.HEADER_ROW));
  }

  /**
   * renames the attribute at the given col index
   * 
   * @param columnIndex the index of the column
   * @param newName the new name of the attribute
   */
  public void renameAttributeAt(int columnIndex, String newName) {
    if (isAttribute(columnIndex)) {
      addUndoPoint();
      try {
        m_Data.renameAttribute(getAttributeIndex(columnIndex), newName);
      } catch (Exception e) {
        ComponentHelper.showInformationBox(null, "Error renaming attribute... ", e.toString(),
                JOptionPane.ERROR_MESSAGE);
        System.out.println(e);
        undo();
      }
      notifyListener(new TableModelEvent(this, TableModelEvent.HEADER_ROW));
    }
  }

  /**
   * set the attribute weight at the given col index
   *
   * @param columnIndex the index of the column
   * @param weight the new weight of the attribute
   */
  public void setAttributeWeightAt(int columnIndex, double weight) {
    if (isAttribute(columnIndex)) {
      addUndoPoint();
      m_Data.setAttributeWeight(getAttributeIndex(columnIndex), weight);
      if (weight != 1.0) {
        m_ShowAttributeWeights = true;
      } else {
        m_ShowAttributeWeights = false;
        for (int i = 0; i < m_Data.numAttributes(); i++) {
          if (m_Data.attribute(i).weight() != 1.0) {
            m_ShowAttributeWeights = true;
            break;
          }
        }
      }
      notifyListener(new TableModelEvent(this, TableModelEvent.HEADER_ROW));
    }
  }

  /**
   * sets the attribute at the given col index as the new class attribute, i.e.
   * it moves it to the end of the attributes
   * 
   * @param columnIndex the index of the column
   */
  public void attributeAsClassAt(int columnIndex) {
    Reorder reorder;
    String order;
    int i;

    if (isAttribute(columnIndex)) {
      addUndoPoint();

      try {
        // build order string (1-based!)
        order = "";
        for (i = 1; i < m_Data.numAttributes() + 1; i++) {
          // skip new class
          if (i - 1 == getAttributeIndex(columnIndex)){
            continue;
          }

          if (!order.equals("")) {
            order += ",";
          }
          order += Integer.toString(i);
        }
        if (!order.equals("")) {
          order += ",";
        }
        order += Integer.toString(getAttributeIndex(columnIndex) + 1);

        // process data
        reorder = new Reorder();
        reorder.setAttributeIndices(order);
        reorder.setInputFormat(m_Data);
        m_Data = Filter.useFilter(m_Data, reorder);

        // set class index
        m_Data.setClassIndex(m_Data.numAttributes() - 1);
      } catch (Exception e) {
        ComponentHelper.showInformationBox(null, "Error setting class attribute... ", e.toString(),
                JOptionPane.ERROR_MESSAGE);
        System.out.println(e);
        undo();
      }

      notifyListener(new TableModelEvent(this, TableModelEvent.HEADER_ROW));
    }
  }

  /**
   * deletes the instance at the given index
   * 
   * @param rowIndex the index of the row
   */
  public void deleteInstanceAt(int rowIndex) {
    deleteInstanceAt(rowIndex, true);
  }

  /**
   * deletes the instance at the given index
   * 
   * @param rowIndex the index of the row
   * @param notify whether to notify the listeners
   */
  public void deleteInstanceAt(int rowIndex, boolean notify) {
    if ((rowIndex >= 0) && (rowIndex < getRowCount())) {
      if (!m_IgnoreChanges) {
        addUndoPoint();
      }
      m_Data.delete(rowIndex);
      if (notify) {
        notifyListener(new TableModelEvent(this, rowIndex, rowIndex,
          TableModelEvent.ALL_COLUMNS, TableModelEvent.DELETE));
      }
    }
  }

  public void setInstanceWeight(int index, double weight) {
    setInstanceWeight(index, weight,true);
  }

  public void setInstanceWeight(int index, double weight, boolean notify) {
    if (!m_IgnoreChanges) {
      addUndoPoint();
    }
    m_Data.instance(index).setWeight(weight);
    if (notify) {
      if (m_ShowInstanceWeights) {
        notifyListener(new TableModelEvent(this, index, 1));
      } else {
        if (weight != 1.0) {
          m_ShowInstanceWeights = true;
          notifyListener(new TableModelEvent(this, TableModelEvent.HEADER_ROW));
        }
      }
    }
  }

  public void insertInstance(int index) {
    insertInstance(index, true);
  }

  public void insertInstance(int index, boolean notify) {
    if (!m_IgnoreChanges) {
      addUndoPoint();
    }
    double[] vals = new double[m_Data.numAttributes()];

    // set any string or relational attribute values to missing
    // in the new instance, just in case this is the very first
    // instance in the dataset.
    for (int i = 0; i < m_Data.numAttributes(); i++) {
      if (m_Data.attribute(i).isString()
        || m_Data.attribute(i).isRelationValued()) {
        vals[i] = Utils.missingValue();
      }
    }
    Instance toAdd = new DenseInstance(1.0, vals);
    if (index < 0) {
      m_Data.add(toAdd);
    } else {
      m_Data.add(index, toAdd);
    }
    if (notify) {
      notifyListener(new TableModelEvent(this, m_Data.numInstances() - 1,
        m_Data.numInstances() - 1, TableModelEvent.ALL_COLUMNS,
        TableModelEvent.INSERT));
    }
  }

  /**
   * deletes the instances at the given positions
   * 
   * @param rowIndices the indices to delete
   */
  public void deleteInstances(int[] rowIndices) {
    int i;

    Arrays.sort(rowIndices);

    addUndoPoint();

    m_IgnoreChanges = true;
    for (i = rowIndices.length - 1; i >= 0; i--) {
      deleteInstanceAt(rowIndices[i], false);
    }
    m_IgnoreChanges = false;

    notifyListener(new TableModelEvent(this, rowIndices[0],
      rowIndices[rowIndices.length - 1], TableModelEvent.ALL_COLUMNS,
      TableModelEvent.DELETE));
  }

  /**
   * sorts the instances via the given attribute (ascending)
   * 
   * @param columnIndex the index of the column
   */
  public void sortInstances(int columnIndex) {
    sortInstances(columnIndex, true);
  }

  /**
   * sorts the instances via the given attribute
   * 
   * @param columnIndex the index of the column
   * @param ascending ascending if true, otherwise descending
   */
  public void sortInstances(int columnIndex, boolean ascending) {
    if (isAttribute(columnIndex) || (m_ShowInstanceWeights && columnIndex == 1)) {
      addUndoPoint();

      double[] vals = new double[m_Data.numInstances()];
      Instance[] backup = new Instance[vals.length];
      for (int i = 0; i < vals.length; i++) {
        Instance inst = m_Data.instance(i);
        backup[i] = inst;
        vals[i] = isAttribute(columnIndex) ? inst.value(getAttributeIndex(columnIndex)) : inst.weight();
      }
      int[] sortOrder = Utils.stableSort(vals);
      for (int i = 0; i < vals.length; i++) {
        m_Data.set(i, backup[sortOrder[i]]);
      }
      if (!ascending) {
        Instances reversedData = new Instances(m_Data, m_Data.numInstances());
        int i = m_Data.numInstances();
        while (i > 0) {
          i--;
          int equalCount = 1;
          if (isAttribute(columnIndex)) {
            while ((i > 0)
                    && (m_Data.instance(i).value(getAttributeIndex(columnIndex)) == m_Data.instance(
                    i - 1).value(getAttributeIndex(columnIndex)))) {
              equalCount++;
              i--;
            }
          } else {
            while ((i > 0)
                    && (m_Data.instance(i).weight() == m_Data.instance(i - 1).weight())) {
              equalCount++;
              i--;
            }
          }
          int j = 0;
          while (j < equalCount) {
            reversedData.add(m_Data.instance(i + j));
            j++;
          }
        }
        m_Data = reversedData;
      }
      notifyListener(new TableModelEvent(this));
    }
  }

  /**
   * returns the column of the given attribute name, -1 if not found
   * 
   * @param name the name of the attribute
   * @return the column index or -1 if not found
   */
  public int getAttributeColumn(String name) {
    int i;
    int result;

    result = -1;

    for (i = 0; i < m_Data.numAttributes(); i++) {
      if (m_Data.attribute(i).name().equals(name)) {
        result = i + (m_ShowInstanceWeights ? 2 : 1);
        break;
      }
    }

    return result;
  }

  /**
   * returns the most specific superclass for all the cell values in the column
   * (always String)
   * 
   * @param columnIndex the column index
   * @return the class of the column
   */
  @Override
  public Class<?> getColumnClass(int columnIndex) {
    Class<?> result;

    result = null;

    if ((columnIndex >= 0) && (columnIndex < getColumnCount())) {
      if (columnIndex == 0) {
        result = Integer.class;
      } else if ((getType(columnIndex) == Attribute.NUMERIC) || (m_ShowInstanceWeights && (columnIndex == 1))) {
        result = Double.class;
      } else {
        result = String.class; // otherwise no input of "?"!!!
      }
    }

    return result;
  }

  /**
   * returns the number of columns in the model
   * 
   * @return the number of columns
   */
  @Override
  public int getColumnCount() {
    int result;

    result = 1;
    if (m_ShowInstanceWeights) {
      result++;
    }
    if (m_Data != null) {
      result += m_Data.numAttributes();
    }

    return result;
  }

  /**
   * checks whether the column represents the class or not
   * 
   * @param columnIndex the index of the column
   * @return true if the column is the class attribute
   */
  protected boolean isClassIndex(int columnIndex) {
    boolean result;
    int index;

    index = m_Data.classIndex();
    result =
      ((index == -1) && (m_Data.numAttributes() - 1 == getAttributeIndex(columnIndex)))
        || (index == getAttributeIndex(columnIndex));

    return result;
  }

  /**
   * returns the name of the column at columnIndex
   * 
   * @param columnIndex the index of the column
   * @return the name of the column
   */
  @Override
  public String getColumnName(int columnIndex) {
    String result;

    result = "";

    if ((columnIndex >= 0) && (columnIndex < getColumnCount())) {
      if (columnIndex == 0) {
        result = "<html><center>No.<br><font size=\"-2\">&nbsp;";
        if (m_ShowAttributeWeights) {
          result += "</font><br><font size=\"-2\">&nbsp;</font>";
        }
        result += "</center></html>";
      } else if ((columnIndex == 1) && (m_ShowInstanceWeights)) {
        result = "<html><center>Weight<br><font size=\"-2\">&nbsp;";
        if (m_ShowAttributeWeights) {
          result += "</font><br><font size=\"-2\">&nbsp;</font>";
        }
        result += "</center></html>";
      } else {
        if (m_Data != null) {
          if ((getAttributeIndex(columnIndex) < m_Data.numAttributes())) {
            result = "<html><center>";

            // index
            if (m_ShowAttributeIndex) {
              result += (m_ShowInstanceWeights ?  (columnIndex - 1) : columnIndex) + ": ";
            }

            // name
            if (isClassIndex(columnIndex)) {
              result +=
                      "<b>" + m_Data.attribute(getAttributeIndex(columnIndex)).name() + "</b>";
            } else {
              result += m_Data.attribute(getAttributeIndex(columnIndex)).name();
            }

            // attribute type
            switch (getType(columnIndex)) {
              case Attribute.DATE:
                result += "<br><font size=\"-2\">Date</font>";
                break;
              case Attribute.NOMINAL:
                result += "<br><font size=\"-2\">Nominal</font>";
                break;
              case Attribute.STRING:
                result += "<br><font size=\"-2\">String</font>";
                break;
              case Attribute.NUMERIC:
                result += "<br><font size=\"-2\">Numeric</font>";
                break;
              case Attribute.RELATIONAL:
                result += "<br><font size=\"-2\">Relational</font>";
                break;
              default:
                result += "<br><font size=\"-2\">???</font>";
            }

            if (m_ShowAttributeWeights) {
              result += "<br><font size=\"-2\">Weight : " +
                      Utils.doubleToString(m_Data.attribute(getAttributeIndex(columnIndex)).weight(), 3) +
                      "</font>";
            }


            result += "</center></html>";
          }
        }
      }
    }

    return result;
  }

  /**
   * returns the number of rows in the model
   * 
   * @return the number of rows
   */
  @Override
  public int getRowCount() {
    if (m_Data == null) {
      return 0;
    } else {
      return m_Data.numInstances();
    }
  }

  /**
   * checks whether the value at the given position is missing
   * 
   * @param rowIndex the row index
   * @param columnIndex the column index
   * @return true if the value at the position is missing
   */
  public boolean isMissingAt(int rowIndex, int columnIndex) {
    boolean result;

    result = false;

    if ((rowIndex >= 0) && (rowIndex < getRowCount()) && isAttribute(columnIndex)) {
      result = (m_Data.instance(rowIndex).isMissing(getAttributeIndex(columnIndex)));
    }

    return result;
  }

  /**
   * returns the double value of the underlying Instances object at the given
   * position, -1 if out of bounds
   * 
   * @param rowIndex the row index
   * @param columnIndex the column index
   * @return the underlying value in the Instances object
   */
  public double getInstancesValueAt(int rowIndex, int columnIndex) {
    double result;

    result = -1;

    if ((rowIndex >= 0) && (rowIndex < getRowCount()) && isAttribute(columnIndex)) {
      result = m_Data.instance(rowIndex).value(getAttributeIndex(columnIndex));
    }

    return result;
  }

  /**
   * returns the value for the cell at columnIndex and rowIndex
   * 
   * @param rowIndex the row index
   * @param columnIndex the column index
   * @return the value at the position
   */
  @Override
  public Object getValueAt(int rowIndex, int columnIndex) {
    Object result;
    String tmp;
    String key;
    boolean modified;

    result = null;
    key = rowIndex + "-" + columnIndex;

    if ((rowIndex >= 0) && (rowIndex < getRowCount()) && (columnIndex >= 0) && (columnIndex < getColumnCount())) {
      if (columnIndex == 0) {
        result = new Integer(rowIndex + 1);
      } else if ((columnIndex == 1) && (m_ShowInstanceWeights)) {
        result = new Double(m_Data.instance(rowIndex).weight());
      } else {
        if (isMissingAt(rowIndex, columnIndex)) {
          result = null;
        } else {
          if (m_Cache.containsKey(key)) {
            result = m_Cache.get(key);
          } else {
            switch (getType(columnIndex)) {
            case Attribute.DATE:
            case Attribute.NOMINAL:
            case Attribute.STRING:
            case Attribute.RELATIONAL:
              result = m_Data.instance(rowIndex).stringValue(getAttributeIndex(columnIndex));
              break;
            case Attribute.NUMERIC:
              result =
                new Double(m_Data.instance(rowIndex).value(getAttributeIndex(columnIndex)));
              break;
            default:
              result = "-can't display-";
            }

            if (getType(columnIndex) != Attribute.NUMERIC) {
              if (result != null) {
                tmp = result.toString();
                modified = false;
                // fix html tags, otherwise Java parser hangs
                if ((tmp.indexOf('<') > -1) || (tmp.indexOf('>') > -1)) {
                  tmp = tmp.replace("<", "(");
                  tmp = tmp.replace(">", ")");
                  modified = true;
                }
                // does it contain "\n" or "\r"? -> replace with red html tag
                if ((tmp.indexOf("\n") > -1) || (tmp.indexOf("\r") > -1)) {
                  tmp =
                    tmp.replaceAll("\\r\\n",
                      "<font color=\"red\"><b>\\\\r\\\\n</b></font>");
                  tmp =
                    tmp.replaceAll("\\r",
                      "<font color=\"red\"><b>\\\\r</b></font>");
                  tmp =
                    tmp.replaceAll("\\n",
                      "<font color=\"red\"><b>\\\\n</b></font>");
                  tmp = "<html>" + tmp + "</html>";
                  modified = true;
                }
                result = tmp;
                if (modified) {
                  m_Cache.put(key, tmp);
                }
              }
            }
          }
        }
      }
    }

    return result;
  }

  /**
   * returns true if the cell at rowindex and columnindexis editable
   * 
   * @param rowIndex the index of the row
   * @param columnIndex the index of the column
   * @return true if the cell is editable
   */
  @Override
  public boolean isCellEditable(int rowIndex, int columnIndex) {
    return ((m_ShowInstanceWeights && (columnIndex == 1)) || isAttribute(columnIndex)) && !isReadOnly();
  }

  /**
   * sets the value in the cell at columnIndex and rowIndex to aValue. but only
   * the value and the value can be changed
   * 
   * @param aValue the new value
   * @param rowIndex the row index
   * @param columnIndex the column index
   */
  @Override
  public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
    setValueAt(aValue, rowIndex, columnIndex, true);
  }

  /**
   * sets the value in the cell at columnIndex and rowIndex to aValue. but only
   * the value and the value can be changed
   * 
   * @param aValue the new value
   * @param rowIndex the row index
   * @param columnIndex the column index
   * @param notify whether to notify the listeners
   */
  public void setValueAt(Object aValue, int rowIndex, int columnIndex, boolean notify) {

    int type;
    int index;
    String tmp;
    Instance inst;
    Attribute att;
    Object oldValue;

    if (!m_IgnoreChanges) {
      addUndoPoint();
    }

    // Is this actually an instance weight?
    if (m_ShowInstanceWeights && columnIndex == 1) {
      double oldWeight = m_Data.instance(rowIndex).weight();
      try {
        double newWeight = Double.parseDouble(aValue.toString());
        if (oldWeight != newWeight) {
          m_Data.instance(rowIndex).setWeight(newWeight);
        } else {
          return;
        }
      } catch (Exception e) {
        // Ignore
      }
      if (notify) {
        notifyListener(new TableModelEvent(this, rowIndex, columnIndex));
      }
      return;
    }

    oldValue = getValueAt(rowIndex, columnIndex);
    type = getType(rowIndex, columnIndex);
    index = getAttributeIndex(columnIndex);
    inst = m_Data.instance(rowIndex);
    att = inst.attribute(index);

    // missing?
    if (aValue == null) {
      inst.setValue(index, Utils.missingValue());
    } else {
      tmp = aValue.toString();

      switch (type) {
      case Attribute.DATE:
        try {
          att.parseDate(tmp);
          inst.setValue(index, att.parseDate(tmp));
        } catch (Exception e) {
          // Ignore
        }
        break;

      case Attribute.NOMINAL:
        if (att.indexOfValue(tmp) > -1) {
          inst.setValue(index, att.indexOfValue(tmp));
        } else {
          // Ignore
        }
        break;

      case Attribute.STRING:
        inst.setValue(index, tmp);
        break;

      case Attribute.NUMERIC:
        try {
          Double.parseDouble(tmp);
          inst.setValue(index, Double.parseDouble(tmp));
        } catch (Exception e) {
          // Ignore
        }
        break;

      case Attribute.RELATIONAL:
        try {
          inst.setValue(index, inst.attribute(index).addRelation((Instances) aValue));
        } catch (Exception e) {
          // Ignore
        }
        break;

      default:
        throw new IllegalArgumentException("Unsupported Attribute type: "
          + type + "!");
      }
    }

    // notify only if the value has changed!
    if (notify && (!("" + oldValue).equals("" + aValue))) {
      notifyListener(new TableModelEvent(this, rowIndex, columnIndex));
    }
  }

  /**
   * adds a listener to the list that is notified each time a change to data
   * model occurs
   * 
   * @param l the listener to add
   */
  @Override
  public void addTableModelListener(TableModelListener l) {
    m_Listeners.add(l);
  }

  /**
   * removes a listener from the list that is notified each time a change to the
   * data model occurs
   * 
   * @param l the listener to remove
   */
  @Override
  public void removeTableModelListener(TableModelListener l) {
    m_Listeners.remove(l);
  }

  /**
   * notfies all listener of the change of the model
   * 
   * @param e the event to send to the listeners
   */
  public void notifyListener(TableModelEvent e) {
    Iterator<TableModelListener> iter;
    TableModelListener l;

    // is notification enabled?
    if (!isNotificationEnabled()) {
      return;
    }

    iter = m_Listeners.iterator();
    while (iter.hasNext()) {
      l = iter.next();
      l.tableChanged(e);
    }
  }

  /**
   * removes the undo history
   */
  @Override
  public void clearUndo() {
    m_UndoList = new Vector<File>();
  }

  /**
   * returns whether an undo is possible, i.e. whether there are any undo points
   * saved so far
   * 
   * @return returns TRUE if there is an undo possible
   */
  @Override
  public boolean canUndo() {
    return !m_UndoList.isEmpty();
  }

  /**
   * undoes the last action
   */
  @Override
  public void undo() {
    File tempFile;
    Instances inst;
    ObjectInputStream ooi;

    if (canUndo()) {
      // load file
      tempFile = m_UndoList.get(m_UndoList.size() - 1);
      try {
        // read serialized data
        ooi =
          new ObjectInputStream(new BufferedInputStream(new FileInputStream(
            tempFile)));
        inst = (Instances) ooi.readObject();
        ooi.close();

        // set instances
        setInstances(inst);
        notifyListener(new TableModelEvent(this, TableModelEvent.HEADER_ROW));
        notifyListener(new TableModelEvent(this));
      } catch (Exception e) {
        ComponentHelper.showInformationBox(null, "Error undoing... ", e.toString(),
                JOptionPane.ERROR_MESSAGE);
        System.out.println(e);
      }
      tempFile.delete();

      // remove from undo
      m_UndoList.remove(m_UndoList.size() - 1);
    }
  }

  /**
   * adds an undo point to the undo history, if the undo support is enabled
   * 
   * @see #isUndoEnabled()
   * @see #setUndoEnabled(boolean)
   */
  @Override
  public void addUndoPoint() {
    File tempFile;
    ObjectOutputStream oos;

    // undo support currently on?
    if (!isUndoEnabled()) {
      return;
    }

    if (getInstances() != null) {
      try {
        // temp. filename
        tempFile = File.createTempFile("arffviewer", null);
        tempFile.deleteOnExit();

        // serialize instances
        oos =
          new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(
            tempFile)));
        oos.writeObject(getInstances());
        oos.flush();
        oos.close();

        // add to undo list
        m_UndoList.add(tempFile);
      } catch (Exception e) {
        ComponentHelper.showInformationBox(null, "Error setting undo point... ", e.toString(),
                JOptionPane.ERROR_MESSAGE);
        System.out.println(e);
      }
    }
  }

  /**
   * Sets whether to display the attribute index in the header.
   * 
   * @param value if true then the attribute indices are displayed in the table
   *          header
   */
  public void setShowAttributeIndex(boolean value) {
    m_ShowAttributeIndex = value;
    fireTableStructureChanged();
  }

  /**
   * Returns whether to display the attribute index in the header.
   * 
   * @return true if the attribute indices are displayed in the table header
   */
  public boolean getShowAttributeIndex() {
    return m_ShowAttributeIndex;
  }
}
