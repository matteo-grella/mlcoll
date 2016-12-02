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

with Ada.Unchecked_Deallocation;

package MLColl.Encoders.BiRNN is

    type RNN_Direction_Type is
      (L2R,  -- Left to Right
       R2L); -- Right to Left

    type Model_Type is abstract tagged record
        Input_Layer_Size  : Positive;
        -- Number of Neuron in the Input Layer

        Hidden_Layer_Size : Positive := 1;
        -- Number of Neuron in the Hidden Layer

        Output_Layer_Size : Positive := 1;
        -- Elements 0 to N/2     : Left to Right
        -- Elements N/2 + 1 to N : Right to Left

        Initialized   : Boolean := False;
    end record;

    type Model_Access_Type is access Model_Type'Class;

    type BiRNN_Structure_Type
      (First_Sequence_Index        : Index_Type;
       Last_Sequence_Index         : Index_Type;
       First_Hidden_Sequence_Index : Extended_Index_Type) is abstract tagged record

        Initialized : Boolean := False;

        Input_Sequence : Real_Array_Access_Array
          (First_Sequence_Index .. Last_Sequence_Index) := (others => null);

        Input_Sequence_Gradients                      : Real_Array_Access_Array
          (First_Sequence_Index .. Last_Sequence_Index) := (others => null);

        Encoded_Sequence                              : Real_Array_Access_Array
          (First_Sequence_Index .. Last_Sequence_Index) := (others => null);

        Sequence_Output_Error                         : Real_Array_Access_Array
          (First_Sequence_Index .. Last_Sequence_Index) := (others => null);

    end record;

    type BiRNN_Structure_Access is access BiRNN_Structure_Type'Class;

    procedure Initialize_BiRNN_Model
      (Model             : in out Model_Type;
       Input_Layer_Size  : in Positive;
       Hidden_Layer_Size : in Positive;
       Output_Layer_Size : in Positive;
       Verbose           : in Boolean := False) is abstract;

    procedure Finalize_BiRNN_Model
      (Model : in out Model_Type) is abstract;

    procedure Initialize_BiRNN_Structure
      (BiRNN_Structure   : in out BiRNN_Structure_Type;
       Input_Layer_Size  : in Positive;
       Hidden_Layer_Size : in Positive;
       Output_Layer_Size : in Positive;
       Input_Sequence    : in Encoded_Entry_Array_Type) is abstract;

    procedure Finalize_BiRNN_Structure
      (BiRNN_Structure  : in out BiRNN_Structure_Type) is abstract;

    procedure Encode
      (Model           : in     Model_Type;
       BiRNN_Structure : in out BiRNN_Structure_Type'Class;
       Verbose         : in     Boolean := False) is abstract;

    procedure Learn
      (Model           : in out Model_Type;
       BiRNN_Structure : in out BiRNN_Structure_Type'Class;
       Verbose         : in     Boolean := False) is abstract;

    procedure Free is new
      Ada.Unchecked_Deallocation
        (Model_Type'Class, Model_Access_Type);

    procedure Free is new
      Ada.Unchecked_Deallocation
        (BiRNN_Structure_Type'Class, BiRNN_Structure_Access);

    type Sequence_Info_Type is record
        First_Index : Extended_Index_Type := -1;
        Last_Index  : Extended_Index_Type := -1;
        Length      : Extended_Index_Type := -1;
    end record;

    procedure Partition
      (V_In   : in Real_Array_Access_Array;
       V1_Out : in Real_Array_Access_Array;
       V2_Out : in Real_Array_Access_Array) with
      Pre => V1_Out (V1_Out'First)'Length = V_In (V_In'First)'Length / 2;

    procedure Concatenate
      (V_Out  : in out Real_Array_Access_Array;
       V1_In  : in     Real_Array_Access_Array;
       V2_In  : in     Real_Array_Access_Array);

    procedure Merge
      (V_Out  : in out Real_Array_Access_Array;
       V1_In  : in     Real_Array_Access_Array;
       V2_In  : in     Real_Array_Access_Array);

    procedure Fill
      (V_In       : in     Real_Array_Access_Array;
       V1_Out     : in out Real_Array_Access_Array;
       V2_Rev_Out : in out Real_Array_Access_Array);

    Encoder_BiRNN_Exception : exception;

end MLColl.Encoders.BiRNN;
