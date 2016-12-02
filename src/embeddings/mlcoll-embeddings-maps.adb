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

with Ada.Text_IO; use Ada.Text_IO;

package body MLColl.Embeddings.Maps is

    procedure Free
      (Key_Vector : in out Key_Vectors.Vector) is
    begin
        for I in Key_Vector.First_Index .. Key_Vector.Last_Index loop
            Free(Key_Vector(I));
        end loop;
    end Free;
    
    procedure Set_Embeddings_Layer_Size
      (Embeddings_Map : in out Embeddings_Map_Type;
       Layer_Size     : in     Length_Type) is
    begin
        Embeddings_Map.Embeddings_Layer_Size := Layer_Size;
    end Set_Embeddings_Layer_Size;
    
    procedure Set_Vocabulary
      (Embeddings_Map : in out Embeddings_Map_Type;
       Elements       : in String_Vectors.Vector) is
    begin
        
        if not Embeddings_Map.Vocabulary.Is_Empty then
            raise Constraint_Error with "Vocabulary already set.";
        end if;
        
        -- Insert special elements
        Embeddings_Map.Vocabulary.Insert (Unknown_Item);
        Embeddings_Map.Vocabulary.Insert(Null_Item);
        
        for Element of Elements loop
            if not Embeddings_Map.Vocabulary.Contains_Element(Element) then
                Embeddings_Map.Vocabulary.Insert(Element);
            end if;
        end loop;
    end;
    
    procedure Initialize
      (Embeddings_Map : in out Embeddings_Map_Type;
       Elements       : in     String_Vectors.Vector;
       Layer_Size     : in     Length_Type;
       Verbose        : in     Boolean := False) is
    begin
        Embeddings_Map.Set_Embeddings_Layer_Size (Layer_Size);
        Embeddings_Map.Set_Vocabulary (Elements);
        Initialize(Embeddings_Map, Verbose);
    end Initialize;
       
    procedure Initialize
      (Embeddings_Map : in out Embeddings_Map_Type;
       Verbose        : in     Boolean := False) is
    begin
        
        Embeddings_Map.Embeddings :=
          New_Embeddings_Structure (Layer_Size => Embeddings_Map.Embeddings_Layer_Size);
              
        Embeddings_Map.Embeddings.Set_Vocabulary_Size
          (Embeddings_Map.Vocabulary.Length);
        
        Embeddings_Map.Embeddings.Create_Matrix 
          (Random_Range => Embeddings_Map.Embeddings_Random_Range);

        if Verbose then
            Put_Line (Standard_Error, "Embeddings dimension ... " -- TODO: Name
                      & Embeddings_Map.Embeddings.Get_Vocabulary_Size'Img
                      & " x "
                      & Embeddings_Map.Embeddings.Get_Layer_Size'Img);
        end if;
        
    end Initialize;

    procedure Look_Up_Null_Item
      (Embeddings_Map  : in     Embeddings_Map_Type;
       Out_Embedding   : in out Real_Access_Array;
       Offset          : in out Index_Type) is
    begin
        Look_Up
          (Embeddings_Map  => Embeddings_Map,
           Key_Label       => Null_Item,
           Try_Lowercase   => False,
           Out_Embedding   => Out_Embedding,
           Offset          => Offset);
    end Look_Up_Null_Item;
    
    procedure Look_Up
      (Embeddings_Map  : in     Embeddings_Map_Type;
       Key_Label       : in     String;
       Try_Lowercase   : in     Boolean := False;
       Out_Embedding   : in out Real_Access_Array;
       Offset          : in out Index_Type) is

        function Get_ID return Extended_Index_Type is
        begin
            return ID : Extended_Index_Type do
                -- exact match first
                ID := Embeddings_Map.Vocabulary.Find_ID (Key_Label);
            
                if ID = -1 and then Try_Lowercase then
                    -- try lower case
                    ID := Embeddings_Map.Vocabulary.Find_ID (String_To_Lower (Key_Label));
                end if;
                  
                if ID = -1 then
                    -- use unknown
                    ID := Embeddings_Map.Vocabulary.Find_ID (Unknown_Item);
                end if; 
            end return;
        end Get_ID;
        
        Embeddings_Matrix : Real_Matrix_Access 
          := Embeddings_Map.Embeddings.Get_Matrix; -- Access!
    begin
          
        for J in Embeddings_Matrix'First (2) .. Embeddings_Matrix'Last (2)loop
            Out_Embedding (Offset) := Embeddings_Matrix(Get_ID, J)'Access;
            Offset := Offset + 1;
        end loop;
   
    end Look_Up;
    
    procedure Propagate_Errors
      (Reference_Map            : in Real_Reference_Maps.Map;
       Learning_Rate            : in Real;
       Regularization_Parameter : in Real := 0.0;
       Class                    : in Index_Type_Array := (Index_Type'First => 0)) is
    begin

        for Reference : Real_Reference_Type of Reference_Map loop
            if Reference.Reference_Access /= null 
              and then (for some C of Class => Reference.Class = C) then
                declare
                    Regularized_Error : constant Real 
                      := Reference.Value 
                        + (Regularization_Parameter * Reference.Reference_Access.all);
                begin
                    Reference.Reference_Access.all 
                      := Reference.Reference_Access.all - (Learning_Rate * Regularized_Error);
                end;
            end if;
        end loop;
    end Propagate_Errors;
    
end MLColl.Embeddings.Maps;
