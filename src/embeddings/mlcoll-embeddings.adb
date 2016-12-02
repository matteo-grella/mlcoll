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

with ARColl.Numerics.Reals.Functions; use ARColl.Numerics.Reals.Functions;

package body MLColl.Embeddings is

    function New_Embeddings_Structure
      (Layer_Size          : in     Length_Type := 0;
       Vocabulary_Size     : in     Length_Type := 0)
       return Embeddings_Structure_Type is
        
    begin
        return Embeddings_Structure : Embeddings_Structure_Type do
            if Layer_Size /= 0 then
                Embeddings_Structure.Set_Layer_Size (Layer_Size);
            end if;
            
            if Vocabulary_Size /= 0 then
                Embeddings_Structure.Set_Vocabulary_Size (Vocabulary_Size);
            end if;
        
            if Layer_Size /= 0 and then Vocabulary_Size /= 0 then
                Embeddings_Structure.Create_Matrix;
            end if;
        end return;
    end New_Embeddings_Structure;
    
    procedure Initialize
      (Embeddings_Structure : in out Embeddings_Structure_Type;
       Layer_Size           : in     Positive_Length_Type;
       Vocabulary_Size      : in     Positive_Length_Type) is
    begin
        Embeddings_Structure.Set_Layer_Size (Layer_Size);
        Embeddings_Structure.Set_Vocabulary_Size (Vocabulary_Size);
        Embeddings_Structure.Create_Matrix;
    end Initialize;
    
    procedure Create_Matrix
      (Embeddings_Structure : in out Embeddings_Structure_Type;
       Random_Range         : in     Real := 0.0) is
    begin
        if Embeddings_Structure.Layer_Size < 1 then
            raise Embeddings_Error with "Invalid Layer Size " & Embeddings_Structure.Layer_Size'Img;
        elsif Embeddings_Structure.Vocabulary_Size < 1 then
            raise Embeddings_Error with "Invalid Vocabulary Size " & Embeddings_Structure.Vocabulary_Size'Img;
        elsif Embeddings_Structure.Has_Matrix then
            raise Embeddings_Error with "Matrix already exists";
        end if;
        
        Embeddings_Structure.Matrix := new Real_Matrix
          (Index_Type'First .. Index_Type'First + Index_Type (Embeddings_Structure.Vocabulary_Size) - 1,
           Index_Type'First .. Index_Type'First + Index_Type (Embeddings_Structure.Layer_Size) - 1);
        
        for I in Embeddings_Structure.Matrix'Range (1) loop
            for J in Embeddings_Structure.Matrix'Range (2) loop
                Embeddings_Structure.Matrix (I, J) 
                  := (if Random_Range /= 0.0 then
                          Get_Random_Weight (Random_Range)
                      else
                          0.0);
            end loop;
        end loop;
        
    end Create_Matrix;
    
    procedure Set_Layer_Size
      (Embeddings_Structure : in out Embeddings_Structure_Type;
       Layer_Size           : in     Positive_Length_Type) is
    begin
        if Embeddings_Structure.Layer_Size_Is_Set then
            raise Embeddings_Error with "Layer Size already set";
        elsif Embeddings_Structure.Has_Matrix then
            raise Embeddings_Error with "Matrix already exists";
        end if;

        Embeddings_Structure.Layer_Size := Layer_Size;
    end Set_Layer_Size;
    
    function Get_Layer_Size
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Length_Type is
    begin
        return Embeddings_Structure.Layer_Size;
    end Get_Layer_Size;
    
    function Layer_Size_Is_Set
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Boolean is
    begin
        return Embeddings_Structure.Layer_Size /= 0;
    end Layer_Size_Is_Set;
    
    procedure Set_Vocabulary_Size
      (Embeddings_Structure : in out Embeddings_Structure_Type;
       Vocabulary_Size      : in     Positive_Length_Type) is
    begin
        if Embeddings_Structure.Vocabulary_Size_Is_Set then
            raise Embeddings_Error with "Vocabulary Size already set";
        elsif Embeddings_Structure.Has_Matrix then
            raise Embeddings_Error with "Matrix already exists";
        end if;

        Embeddings_Structure.Vocabulary_Size := Vocabulary_Size;
    end Set_Vocabulary_Size;
    
    function Get_Vocabulary_Size
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Length_Type is
    begin
        return Embeddings_Structure.Vocabulary_Size;
    end Get_Vocabulary_Size;
    
    function Vocabulary_Size_Is_Set
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Boolean is
    begin
        return Embeddings_Structure.Vocabulary_Size /= 0;
    end Vocabulary_Size_Is_Set;
    
    function Has_Matrix
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Boolean is
    begin
        return Embeddings_Structure.Matrix /= null;
    end Has_Matrix;
    
    function Get_Matrix
      (Embeddings_Structure : Embeddings_Structure_Type)
       return Real_Matrix_Access is
    begin
        if not Embeddings_Structure.Has_Matrix then
            raise Embeddings_Error with "Matrix does not exist";
        end if;

        return Embeddings_Structure.Matrix;
    end Get_Matrix;
    
--      procedure Read_Embed_File
--        (Embed_Filename       : in     String;
--         WE_Model             : in out Word_Embeddings_Model_Type;
--         Transformation       : in     Embeddings_Transformation_Type := SCALING;
--         Fixed_Embedding_Size : in     Positive) is
--  
--          File      : File_Type;
--          Lines     : String_Vectors.Vector;
--  
--          Num_Words      : Natural := 0;
--          Embedding_Size : Natural := 0;
--            
--          use GNAT;
--          Slices         : String_Split.Slice_Set;
--  
--          Embeddings_Indexes  : String_To_Index_Maps.Map renames WE_Model.Embeddings_Indexes;
--          Embeddings     : Float_Matrix_Access_Type   renames WE_Model.Embeddings;
--      begin
--          if Embeddings /= null then
--              raise Word_Embeddings_Exception with "not null Embeddings";
--          end if;
--  
--          -- Read Embed File and fill Lines
--  
--          Open (File, In_File, Embed_Filename);
--  
--          while not End_Of_File (File) loop
--              declare
--                  Line : constant String := Get_Line (File);
--              begin
--                  if Line'Length > 0 then
--                      Lines.Append ((if Line(Line'Last) = ' ' then Line(Line'First .. Line'Last - 1) else Line));
--                  end if;
--              end;
--          end loop;
--  
--          Close (File);
--  
--          -- Check embedding dimensions
--  
--          Num_Words := Natural (Lines.Length);
--  
--          String_Split.Create
--            (Slices, Lines.First_Element, " ", String_Split.Multiple);
--  
--          Embedding_Size := Natural (String_Split.Slice_Count (Slices)) - 1;
--  
--          Put_Line ("Embedding File " & Embed_Filename);
--          Put_Line ("    numWords =" & Num_Words'Img &
--                      ", embeddingSize =" & Embedding_Size'Img);
--  
--          if Embedding_Size /= Fixed_Embedding_Size then
--              Put_Line("WARNING: Word_Embeddings_Exception: The dimension of embedding file does not match Fixed_Embedding_Size");
--              Embedding_Size := Fixed_Embedding_Size;
--          end if;
--  
--          --WE_Model.Embedding_Size := Embedding_Size;
--            
--          -- Load embeddings values from Lines
--  
--          Embeddings := new Float_Matrix_Type
--            (Index_Type'First
--             ..
--               Index_Type'First + Index_Type(Num_Words-1),
--  
--             Index_Type'First
--             ..
--               Index_Type'First + Index_Type(Embedding_Size-1));
--          
--  
--          Put_Line ("Embedding Matrix: " & Embeddings'Length(1)'Img & " X " & Embeddings'Length(2)'Img);
--          
--          Embeddings_Indexes.Clear;
--  
--          for I in Embeddings'Range (1) loop
--  
--              String_Split.Create
--                (Slices, Lines.Element (I), " ", String_Split.Multiple);
--  
--              Embeddings_Indexes.Insert (String_Split.Slice (Slices, 1), I);
--              --Put_Line(String_Split.Slice (Slices, 1));
--              
--              for J in Embeddings'Range (2) loop
--                  Embeddings (I, J) := Real'Value
--                    (String_Split.Slice
--                       (Slices, String_Split.Slice_Number (J - 1 + 2)));
--              end loop;
--    
--          end loop;
--  
--  
--          if Transformation = SCALING then
--              Scaling(Embeddings);        
--              
--          elsif Transformation = BINARIZE then  
--              for I in Embeddings'Range (1) loop
--                  for J in Embeddings'Range (2) loop
--                      Embeddings (I, J) := (if Embeddings (I, J) > 0.0 then 1.0 else 0.0);
--                  end loop;
--              end loop;
--          end if;
--          
--      end Read_Embed_File;
--      
--      
    overriding procedure Finalize
      (Embeddings_Structure : in out Embeddings_Structure_Type) is
    begin
        Free (Embeddings_Structure.Matrix);

        Embeddings_Structure.Layer_Size      := 0;
        Embeddings_Structure.Vocabulary_Size := 0;
    end Finalize;
    
    overriding procedure Adjust
      (Embeddings_Structure : in out Embeddings_Structure_Type) is
    begin
        if Embeddings_Structure.Matrix /= null then
            Embeddings_Structure.Matrix
              := new Real_Matrix'(Embeddings_Structure.Matrix.all);
        end if;
    end Adjust;

end MLColl.Embeddings;
