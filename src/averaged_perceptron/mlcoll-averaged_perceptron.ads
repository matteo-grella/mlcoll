------------------------------------------------------------------------------
--                               M L C O L L
--  M a c h i n e   L e a r n i n g   C o m p o n e n t   C o l l e c t i o n
--
--        Copyright 2009-2013 M. Grella, S. Cangialosi, E. Brambilla
--
--  This is free software; you can redistribute it and/or modify it under
--  terms of the GNU General Public License as published by the Free Software
--  Foundation; either version 2, or (at your option) any later version.
--  This software is distributed in the hope that it will be useful, but WITH
--  OUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
--  or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
--  for more details. Free Software Foundation, 59 Temple Place - Suite
--  330, Boston, MA 02111-1307, USA.
--
--  As a special exception, if other files instantiate generics from this
--  unit, or you link this unit with other files to produce an executable,
--  this unit does not by itself cause the resulting executable to be
--  covered by the GNU General Public License. This exception does not
--  however invalidate any other reasons why the executable file might be
--  covered by the GNU Public License.
--
------------------------------------------------------------------------------

pragma License (Modified_GPL);

with Ada.Containers.Indefinite_Vectors;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

private with Ada.Containers;
private with Ada.Containers.Indefinite_Hashed_Maps;
with Ada.Strings.Hash;

with ARColl.Containers.Sparse_Vectors.Generic_Sparse_Vectors;

generic

    type Index_Type is range <>;
    type Float_Type is digits <>;

package MLColl.Averaged_Perceptron is

    Averaged_Perceptron_Exception : exception;

    type FLOAT_Vector_Type is array (Index_Type range <>) of FLOAT_Type;

    type Model_Type is tagged private;

    type Feature_Value_Type is record
        Feature : Unbounded_String;
        Value   : Float_Type;
    end record;
    -- Feature + Float value

    package Features_Values_Vectors is
      new Ada.Containers.Indefinite_Vectors
        (Index_Type   => Natural,
         Element_Type => Feature_Value_Type);
    -- Vectors of String features + Float values

    function Hash
      (Key : Index_Type)
       return Ada.Containers.Hash_Type is
      (Ada.Containers.Hash_Type (Key))
    with Inline;

    type Feature_Class_Info_Type is record
        Weight     : Float_Type := 0.0;
        -- Weight

        Totals     : Float_Type := 0.0;
        -- The accumulated values, for the averaging.

        Timestamps : Natural    := 0;
        -- The last time the feature was changed, for the averaging.
    end record;

    package Class_To_Value_Sparse_Vectors is new
      ARColl.Containers.Sparse_Vectors.Generic_Sparse_Vectors
        (Index_Type          => Index_Type,
         Element_Type        => Feature_Class_Info_Type,
         Conversion_Treshold => 10,
         "="                 => "=");
    -- Maps Classes to their values

    procedure Initialize
      (Model             : in out Model_Type;
       Min_Update_Cutoff : Natural);

    procedure Finalize
      (Model : in out Model_Type);

    procedure Update
      (Model         : in out Model_Type;
       Correct_Class : in String;
       Guess_Class   : in String;
       Features_Values : in Features_Values_Vectors.Vector);

    procedure Update
      (Model         : in out Model_Type;
       Correct_Class : in String;
       Features_Values : in Features_Values_Vectors.Vector);

    function Score
      (Model        : in Model_Type;
       Features     : in Features_Values_Vectors.Vector)
       return FLOAT_Vector_Type;

    function Predict
      (Model    : Model_Type;
       Features : Features_Values_Vectors.Vector)
       return Index_Type;
    -- Dot-product the features and current weights and return the best class.

    function Get_Outcome_Label
      (Model : in Model_Type;
       Class : in Index_Type) return String;

    procedure Average_Weights
      (Model : in out Model_Type)
      with Inline;

    procedure Serialize
      (Model          : in Model_Type;
       Model_Filename : in String);
    -- Serialize Model to file

    procedure Load
      (Model          : out Model_Type;
       Model_Filename : in  String);
    -- Load Model from serialized file

    procedure Prune_Features
      (Model : in out Model_Type);

    procedure Print_Stats
      (Model : in out Model_Type);

private

    subtype Extended_Index_Type is Index_Type'Base;

    type Update_And_Classes_Type is record
        Update_Count : Natural := 0;
        Classes      : Class_To_Value_Sparse_Vectors.Sparse_Vector_Type;
    end record;

    package Weights_Maps is new
      Ada.Containers.Indefinite_Hashed_Maps
        (Key_Type        => String,
         Element_Type    => Update_And_Classes_Type,
         Hash            => Ada.Strings.Hash,
         Equivalent_Keys => "=");

    package Word_Index_Maps is new
      Ada.Containers.Indefinite_Hashed_Maps
        (Key_Type        => String,
         Element_Type    => Index_Type,
         Hash            => Ada.Strings.Hash,
         Equivalent_Keys => "=");
    -- Associates an ID to each word in a set.

    package Labels_Vectors is new
      Ada.Containers.Indefinite_Vectors
        (Index_Type   => Index_Type,
         Element_Type => String);
    -- Map an ID with the corresponding (string) label


    type Model_Type is tagged record

        Min_Update_Cutoff : Natural := 0;
        -- used in Prune_Features procedure as described
        -- in Learning Sparser Perceptron Models tech-report
        -- (Yoav Goldberg and Michael Elhadad, 2011)

        Num_Of_Classes    : Natural := 0;

        First_Class    : Extended_Index_Type := Extended_Index_Type(Index_Type'First);
        Last_Class     : Extended_Index_Type := Extended_Index_Type(Index_Type'First) - 1;

        Outcome_Labels_Vector : Labels_Vectors.Vector;
        -- Names of outcomes

        Outcome_Index_Map     : Word_Index_Maps.Map;
        -- Associates outcome labels to index

        Weights               : Weights_Maps.Map;
        -- Learned Weights

        Instances_Count       : Natural := 0;

        Initialized : Boolean := False;
    end record;

    procedure Update
      (Model           : in out Model_Type;
       Correct_Class   : in     Index_Type;
       Guess_Class     : in     Extended_Index_Type;
       Features_Values : in     Features_Values_Vectors.Vector)
      with Inline;

end MLColl.Averaged_Perceptron;
