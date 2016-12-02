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

with Ada.Finalization;

with ARColl; use ARColl;
with ARColl.Numerics.Reals; use ARColl.Numerics.Reals;

package MLColl.Embeddings is

    type Embeddings_Structure_Type is tagged private;

    function New_Embeddings_Structure
      (Layer_Size          : in     Length_Type := 0;
       Vocabulary_Size     : in     Length_Type := 0)
       return Embeddings_Structure_Type
      with Inline;
    -- Returns a new Embedding_Structure_Type.
    -- If both Layer_Size and Vocabulary_Size are set, the Matrix is created.

    procedure Initialize
      (Embeddings_Structure : in out Embeddings_Structure_Type;
       Layer_Size           : in     Positive_Length_Type;
       Vocabulary_Size      : in     Positive_Length_Type)
      with Inline;

    procedure Set_Layer_Size
      (Embeddings_Structure : in out Embeddings_Structure_Type;
       Layer_Size           : in     Positive_Length_Type)
      with Inline;

    function Get_Layer_Size
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Length_Type
      with Inline;

    function Layer_Size_Is_Set
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Boolean
      with Inline;

    procedure Set_Vocabulary_Size
      (Embeddings_Structure : in out Embeddings_Structure_Type;
       Vocabulary_Size      : in     Positive_Length_Type)
      with Inline;

    function Get_Vocabulary_Size
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Length_Type
      with Inline;

    function Vocabulary_Size_Is_Set
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Boolean
      with Inline;

    function Has_Matrix
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Boolean
      with Inline;

    procedure Create_Matrix
      (Embeddings_Structure : in out Embeddings_Structure_Type;
       Random_Range         : in     Real := 0.0)
      with Inline;

    function Get_Matrix
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Real_Matrix_Access
      with Inline;

    Embeddings_Error : exception;

private

    type Embeddings_Structure_Type is new Ada.Finalization.Controlled with
        record
            Layer_Size      : Length_Type := 0;
            Vocabulary_Size : Length_Type := 0;
            Matrix          : Real_Matrix_Access := null;
        end record;

    overriding procedure Finalize
      (Embeddings_Structure : in out Embeddings_Structure_Type)
      with Inline;

    overriding procedure Adjust
      (Embeddings_Structure : in out Embeddings_Structure_Type)
      with Inline;

end MLColl.Embeddings;
