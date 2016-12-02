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

with MLColl.Embeddings.Maps; use MLColl.Embeddings.Maps;

package MLColl.Encoders.Embeddings is
    
    function Get_Null_Concatenated_Embeddings
      (Embeddings_Maps     : in Embeddings_Map_Array) return Real_Access_Array;
        
    function Get_Concatenated_Embeddings
      (Embeddings_Keys     : in Embeddings_Key_Array;
       Embeddings_Maps     : in Embeddings_Map_Array) return Real_Access_Array with
      Pre => Embeddings_Keys'Length = Embeddings_Maps'Length;
    
    procedure Encode_Sequence
      (Key_Vector          : in     Key_Vectors.Vector;
       Embeddings_Maps     : in     Embeddings_Map_Array;
       Encoded_Token_Size  : in     Positive;
       Encoded_Sequence    : in out MLColl.Encoders.Encoded_Entry_Array_Type) with 
    Pre => not Key_Vector.Is_Empty and then Key_Vector.First_Element.all'Length = Embeddings_Maps'Length;
    
end MLColl.Encoders.Embeddings;
