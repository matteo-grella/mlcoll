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

with ARColl.Strings; use  ARColl.Strings;
with ARColl.Strings.Unbounded; use  ARColl.Strings.Unbounded;
with ARColl.Containers.Bimaps.String_Id_Bimaps;
with ARColl.Numerics.Reals.References; use ARColl.Numerics.Reals.References;
with Ada.Containers.Vectors;
with Ada.Unchecked_Deallocation;

package MLColl.Embeddings.Maps is

    Unknown_Item   : constant String := "__UNKNOWN__";
    Null_Item      : constant String := "__NULL__";

    type Embeddings_Map_Type is tagged
        record
            Vocabulary              : ARColl.Containers.Bimaps.String_Id_Bimaps.Bimap_Type;
            Embeddings              : Embeddings_Structure_Type;
            Embeddings_Layer_Size   : Length_Type := 1;
            Embeddings_Random_Range : Real := 0.1;
            -- Embeddings Indexes are 1:1 aligned with Vocabulary IDs
        end record;

    type Embeddings_Map_Access is access all Embeddings_Map_Type;

    type Embeddings_Map_Array is array (Index_Type range <>) of Embeddings_Map_Access;

    type Embeddings_Key_Type is
       record
           Key           : Unbounded_String;
           Try_Lowercase : Boolean := False;
       end record;

    type Embeddings_Key_Array is array (Index_Type range <>) of Embeddings_Key_Type;
    type Embeddings_Key_Array_Access is access Embeddings_Key_Array;

    procedure Free is new
      Ada.Unchecked_Deallocation (Embeddings_Key_Array, Embeddings_Key_Array_Access);

    package Key_Vectors is
      new Ada.Containers.Vectors
        (Index_Type   => Index_Type,
         Element_Type => Embeddings_Key_Array_Access);

    procedure Free
      (Key_Vector : in out Key_Vectors.Vector);

    procedure Set_Vocabulary
      (Embeddings_Map : in out Embeddings_Map_Type;
       Elements       : in String_Vectors.Vector);

    procedure Set_Embeddings_Layer_Size
      (Embeddings_Map : in out Embeddings_Map_Type;
       Layer_Size     : in     Length_Type);

    procedure Initialize
      (Embeddings_Map : in out Embeddings_Map_Type;
       Elements       : in     String_Vectors.Vector;
       Layer_Size     : in     Length_Type;
       Verbose        : in     Boolean := False);

    procedure Initialize
      (Embeddings_Map : in out Embeddings_Map_Type;
       Verbose        : in     Boolean := False);

    procedure Look_Up
      (Embeddings_Map  : in     Embeddings_Map_Type;
       Key_Label       : in     String;
       Try_Lowercase   : in     Boolean := False;
       Out_Embedding   : in out Real_Access_Array;
       Offset          : in out Index_Type);

    procedure Look_Up_Null_Item
      (Embeddings_Map  : in     Embeddings_Map_Type;
       Out_Embedding   : in out Real_Access_Array;
       Offset          : in out Index_Type);

    procedure Propagate_Errors
      (Reference_Map            : in Real_Reference_Maps.Map;
       Learning_Rate            : in Real;
       Regularization_Parameter : in Real := 0.0;
       Class                    : in Index_Type_Array := (Index_Type'First => 0));

end MLColl.Embeddings.Maps;
