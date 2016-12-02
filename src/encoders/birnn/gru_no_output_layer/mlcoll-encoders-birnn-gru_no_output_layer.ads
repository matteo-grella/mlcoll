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

with MLColl.Neural_Networks.Recurrent.GRU_No_Output_Layer;

private with Ada.Unchecked_Deallocation;

package MLColl.Encoders.BiRNN.GRU_No_Output_Layer is

    package GRU renames MLColl.Neural_Networks.Recurrent.GRU_No_Output_Layer;

    type GRU_Structure_Access_Type is access GRU.GRU_Structure_Type;

    type RNN_Model_Array     is array (RNN_Direction_Type) of GRU.Model_Type;
    type RNN_Structure_Array is array (RNN_Direction_Type) of GRU_Structure_Access_Type;

    type GRU_Model_Type is new Model_Type with
        record
            RNN_Models : RNN_Model_Array;
        end record;

    type BiRNN_GRU_Structure_Type is new BiRNN_Structure_Type with
        record
            RNN_Structure : RNN_Structure_Array;
        end record;

    overriding procedure Initialize_BiRNN_Model
      (Model             : in out GRU_Model_Type;
       Input_Layer_Size  : in Positive;
       Hidden_Layer_Size : in Positive;
       Output_Layer_Size : in Positive;
       Verbose           : in Boolean := False) with
      Pre => not Model.Initialized
      and then Output_Layer_Size mod 2 = 0
      and then Output_Layer_Size = Hidden_Layer_Size * 2;

    overriding procedure Finalize_BiRNN_Model
      (Model : in out GRU_Model_Type)
      with Pre => Model.Initialized;

    overriding procedure Initialize_BiRNN_Structure
      (BiRNN_Structure   : in out BiRNN_GRU_Structure_Type;
       Input_Layer_Size  : in Positive;
       Hidden_Layer_Size : in Positive;
       Output_Layer_Size : in Positive;
       Input_Sequence    : in     Encoded_Entry_Array_Type) with
      Pre => not BiRNN_Structure.Initialized,
      Post => BiRNN_Structure.Initialized;

    overriding procedure Finalize_BiRNN_Structure
      (BiRNN_Structure  : in out BiRNN_GRU_Structure_Type) with
      Pre => BiRNN_Structure.Initialized,
      Post => not BiRNN_Structure.Initialized;

    overriding procedure Encode
      (Model           : in     GRU_Model_Type;
       BiRNN_Structure : in out BiRNN_Structure_Type'Class;
       Verbose         : in     Boolean := False) with
      Pre => BiRNN_Structure.Initialized and BiRNN_Structure in BiRNN_GRU_Structure_Type;

    overriding procedure Learn
      (Model           : in out GRU_Model_Type;
       BiRNN_Structure : in out BiRNN_Structure_Type'Class;
       Verbose         : in     Boolean := False) with
      Pre => BiRNN_Structure.Initialized and BiRNN_Structure in BiRNN_GRU_Structure_Type;

    type BiRNN_Loss_Array is array (RNN_Direction_Type) of Real;

private

    procedure Free is new
      Ada.Unchecked_Deallocation
        (GRU.GRU_Structure_Type, GRU_Structure_Access_Type);

end MLColl.Encoders.BiRNN.GRU_No_Output_Layer;

