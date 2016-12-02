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

with ARColl; use ARColl;
with ARColl.Strings.Unbounded; use ARColl.Strings.Unbounded;

package body MLColl.Encoders.Embeddings is
    
    function Get_Null_Concatenated_Embeddings
      (Embeddings_Maps : in Embeddings_Map_Array) return Real_Access_Array is
        
        First_Index : constant Index_Type := Index_Type'First;
        Last_Index  : Index_Type := Index_Type'First; 
        
        Offset      : Index_Type := First_Index;
    begin

        for Embeddings_Map of Embeddings_Maps loop
            Last_Index := Last_Index + Index_Type (Embeddings_Map.Embeddings.Get_Layer_Size);
        end loop;
        Last_Index := Last_Index - 1;
        
        return Out_Embedding : Real_Access_Array
          (First_Index .. Last_Index) do
            
            for Embeddings_Map of Embeddings_Maps loop
                Look_Up_Null_Item
                  (Embeddings_Map => Embeddings_Map.all,
                   Out_Embedding  => Out_Embedding,
                   Offset         => Offset);         
            end loop;
            
        end return;
        
    end Get_Null_Concatenated_Embeddings;
    
    function Get_Concatenated_Embeddings
      (Embeddings_Keys     : in Embeddings_Key_Array;
       Embeddings_Maps     : in Embeddings_Map_Array) return Real_Access_Array is
        
        First_Index : constant Index_Type := Index_Type'First;
        Last_Index  : Index_Type := Index_Type'First; 
        
        Offset      : Index_Type := First_Index;
    begin

        for Embeddings_Map of Embeddings_Maps loop
            Last_Index := Last_Index + Index_Type (Embeddings_Map.Embeddings.Get_Layer_Size);
        end loop;
        Last_Index := Last_Index - 1;
        
        return Out_Embedding : Real_Access_Array
          (First_Index .. Last_Index) do
            
            for I in Embeddings_Maps'Range loop
                Look_Up
                  (Embeddings_Map => Embeddings_Maps(I).all,
                   Key_Label      => To_String(Embeddings_Keys(I).Key),
                   Try_Lowercase  => Embeddings_Keys(I).Try_Lowercase,
                   Out_Embedding  => Out_Embedding,
                   Offset         => Offset);    
            end loop;
            
        end return;
        
    end Get_Concatenated_Embeddings;
    
    procedure Encode_Sequence
      (Key_Vector               : in     Key_Vectors.Vector;
       Embeddings_Maps          : in     Embeddings_Map_Array;
       Encoded_Token_Size       : in     Positive;
       Encoded_Sequence         : in out MLColl.Encoders.Encoded_Entry_Array_Type) is
    begin
        
        for T in Encoded_Sequence'Range loop  
            Encoded_Sequence (T) := new Real_Access_Array
              '(Get_Concatenated_Embeddings 
                  (Embeddings_Keys    => Key_Vector(T).all,
                   Embeddings_Maps    => Embeddings_Maps));
                
            pragma Assert (Encoded_Sequence (T)'First = Index_Type'First);
            pragma Assert (Encoded_Sequence (T)'Last  = Index_Type'First + Index_Type (Encoded_Token_Size) - Index_Type (1));
        end loop;
        
    end Encode_Sequence;
    
end MLColl.Encoders.Embeddings;
