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

with Ada.Directories;
with Text_IO;

with ARColl; use ARColl;
with ARColl.Strings; use ARColl.Strings;
with ARColl.IO; use ARColl.IO;
with ARColl.Numerics.Reals; use ARColl.Numerics.Reals;
with ARColl.Readers.JSON;

package body MLColl.Neural_Networks.Datasets.JSON_Loader is

    function Load_Dataset (JSON_Filename : String) return MLColl.Neural_Networks.Datasets.Dataset_Type is
        use MLColl.Neural_Networks.Datasets;
        use ARColl.Readers.JSON;
        
        Dataset : Dataset_Type;
        
        File_Content : String_Access;
        
        JSON_Data : JSON_Element_Class_Access;
    begin
        Text_IO.Put_Line ("-- Loading Dataset from JSON file: "
                          & Ada.Directories.Simple_Name (JSON_Filename) & "...");
        
        File_Content := File_To_String (JSON_Filename);
        JSON_Data := JSON_Decode (File_Content.all);
        
        for JExample of Vector (JSON_Data) loop
            declare
                JFeatures      : constant JSON_Element_Class_Access
                  := Vector (JExample).Element (0);
                
                JOutcome_Index : constant JSON_Element_Class_Access
                  := Vector (JExample).Element (1);

                Example       : Example_Type;
                Feature_Index : Index_Type := Index_Type'First;
            begin
                Example.Outcome_Index
                  := Index_Type'First + Index_Type (Natural'(Value (JOutcome_Index)));
                
                Example.Features := new Real_Array
                  (Index_Type'First .. Index_Type'First + Index_Type (Length (JFeatures)) - 1);
                
                for JFeature of Vector (JFeatures) loop                    
                    Example.Features (Feature_Index) := Real (Float'(Value (JFeature)));
                    Feature_Index := Feature_Index + 1;
                end loop;
                
                Dataset.Append(Example);
            end;
        end loop;
        
        Destroy (JSON_Data);
        Free (File_Content);
        
        Text_IO.Put_Line ("-- Dataset succesfully loaded.");
        
        return Dataset;
        
    exception
        when others =>
            Destroy (JSON_Data);
            Free (File_Content);
            raise;
    end Load_Dataset;
    
end MLColl.Neural_Networks.Datasets.JSON_Loader;

