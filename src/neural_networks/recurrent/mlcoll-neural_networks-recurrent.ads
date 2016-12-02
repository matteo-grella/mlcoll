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

package MLColl.Neural_Networks.Recurrent is
   
    type RNN_Array_Of_Float_Vectors_Type is
      array (Extended_Index_Type range <>) of Real_Array_Access;
    
    procedure Free(Array_Of_Float_Vectors : in RNN_Array_Of_Float_Vectors_Type);
   
    ---
    -- Attention Layers
    ---
    
    type Real_Array_Access_Array_Access is access all Real_Array_Access_Array;
    
    procedure Free is new Ada.Unchecked_Deallocation 
      (Real_Array_Access_Array, Real_Array_Access_Array_Access);
    
    type Attention_Layer_Array is array (Index_Type range <>) of Real_Array_Access_Array_Access;
    type Attention_Layer_Array_Access is access all Attention_Layer_Array;
    
    procedure Free is new
      Ada.Unchecked_Deallocation (Attention_Layer_Array, Attention_Layer_Array_Access);
    
    procedure Free(Array_Of_Array_of_Float_Vectors : in Attention_Layer_Array);
    
end MLColl.Neural_Networks.Recurrent;
