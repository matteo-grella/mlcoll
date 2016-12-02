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

with Text_IO; use Text_IO;
private with Ada.Streams.Stream_IO;

package body MLColl.Averaged_Perceptron is

    package Stream_IO renames Ada.Streams.Stream_IO;

    procedure Initialize
      (Model             : in out Model_Type;
       Min_Update_Cutoff : Natural) is
    begin

        Model.Min_Update_Cutoff := Min_Update_Cutoff;

        Model.Num_Of_Classes := 0;

        Model.First_Class := Extended_Index_Type (Index_Type'First);
        Model.Last_Class := Extended_Index_Type (Index_Type'First) - 1;

        Model.Weights.Clear;
        Model.Weights.Reserve_Capacity (10_000_000);

        -- Number of instances seen
        Model.Instances_Count := 0;

        -- TODO: change this...
        Model.Outcome_Labels_Vector.Append ("__NIL__");
        Model.Outcome_Index_Map.Insert
          (Key      => "__NIL__",
           New_Item => Model.Outcome_Labels_Vector.Last_Index);
        Model.Last_Class :=  Model.Last_Class + 1;

        Model.Num_Of_Classes := Natural (Model.First_Class) + Natural (Model.Last_Class) - 1;

    end Initialize;

    procedure Finalize
      (Model : in out Model_Type) is
    begin
        Model.Num_Of_Classes := 0;

        Model.First_Class := Extended_Index_Type (Index_Type'First);
        Model.Last_Class := Extended_Index_Type (Index_Type'First) - 1;

        Model.Weights.Clear;

        Model.Instances_Count := 0;
    end Finalize;

    function Get_Outcome_Label
      (Model : in Model_Type;
       Class : in Index_Type) return String is
    begin
        return Model.Outcome_Labels_Vector.Element (Class);
    end Get_Outcome_Label;

    procedure Update
      (Model           : in out Model_Type;
       Correct_Class   : in String;
       Guess_Class     : in String;
       Features_Values : in Features_Values_Vectors.Vector) is
    begin

        if not Model.Outcome_Index_Map.Contains (Correct_Class) then
            Model.Outcome_Labels_Vector.Append (Correct_Class);
            Model.Outcome_Index_Map.Insert
              (Key      => Correct_Class,
               New_Item => Model.Outcome_Labels_Vector.Last_Index);
        end if;

        Update
          (Model           => Model,
           Correct_Class   => Model.Outcome_Index_Map.Element (Correct_Class),
           Guess_Class     => Extended_Index_Type(Model.Outcome_Index_Map.Element (Guess_Class)),
           Features_Values => Features_Values);
    end;

    procedure Update
      (Model           : in out Model_Type;
       Correct_Class   : in String;
       Features_Values : in Features_Values_Vectors.Vector) is
    begin

        if not Model.Outcome_Index_Map.Contains (Correct_Class) then
            Model.Outcome_Labels_Vector.Append (Correct_Class);
            Model.Outcome_Index_Map.Insert
              (Key      => Correct_Class,
               New_Item => Model.Outcome_Labels_Vector.Last_Index);
        end if;

        Update
          (Model           => Model,
           Correct_Class   => Model.Outcome_Index_Map.Element (Correct_Class),
           Guess_Class     => -1,
           Features_Values => Features_Values);
    end;

    procedure Update
      (Model           : in out Model_Type;
       Correct_Class   : in     Index_Type;
       Guess_Class     : in     Extended_Index_Type;
       Features_Values : in     Features_Values_Vectors.Vector) is

        procedure Update_Feature
          (Class   : Index_Type;
           Weights : Weights_Maps.Reference_Type;
           Value   : Float_Type)
          with Inline is

            Class_Position : constant Class_To_Value_Sparse_Vectors.Cursor
              := Weights.Element.Classes.Find (Class);

            Class_Position_Has_Element : constant Boolean
              := Class_To_Value_Sparse_Vectors.Has_Element (Class_Position);

        begin

            Weights.Update_Count := Weights.Update_Count + 1;

            if Extended_Index_Type (Class) > Model.Last_Class then
                Model.Last_Class := Extended_Index_Type (Correct_Class);
                Model.Num_Of_Classes := Natural (Model.First_Class) +
                  Natural (Model.Last_Class) - 1;
            end if;

            if Class_Position_Has_Element then

                declare
                    procedure Update_Procedure
                      (Key     : Index_Type;
                       Element : in out Feature_Class_Info_Type)
                      with Inline is
                        pragma Unreferenced (Key);
                    begin
                        Element.Totals := Element.Totals + (Float_Type (Model.Instances_Count - Element.Timestamps) * Element.Weight);
                        Element.Weight := Element.Weight + Value;
                        Element.Timestamps := Model.Instances_Count;
                    end Update_Procedure;
                begin
                    Weights.Classes.Update_Element
                      (Class_Position, Update_Procedure'Access);
                end;

            else
                Weights.Classes.Insert
                  (Class,
                   (Weight     => Value,
                    Totals     => (Float_Type (Model.Instances_Count)),
                    Timestamps => Model.Instances_Count));
            end if;

        end Update_Feature;

    begin

        Model.Instances_Count := Model.Instances_Count + 1;

        if Extended_Index_Type(Correct_Class) /= Guess_Class then

            for Feature_Value of Features_Values loop
                --Put_Line(To_String(Feature_Value.Feature));
                declare

                    Feature : constant String := To_String (Feature_Value.Feature);

                    Feature_Position : Weights_Maps.Cursor
                      := Model.Weights.Find (Feature);

                    Inserted : Boolean;
                begin

                    if not Weights_Maps.Has_Element (Feature_Position) then

                        Model.Weights.Insert
                          (Key      => Feature,
                           New_Item => (Update_Count => 0, Classes => Class_To_Value_Sparse_Vectors.Empty_Sparse_Vector),
                           Position => Feature_Position,
                           Inserted => Inserted);

                        if not Inserted then
                            raise Averaged_Perceptron_Exception
                              with "Insertion failed";
                        end if;
                    end if;


                    Update_Feature
                      (Class   => Correct_Class,
                       Weights => Model.Weights.Reference (Feature_Position),
                       Value   => 1.0);

                    if Guess_Class /= -1 then
                        Update_Feature
                          (Class   => Index_Type(Guess_Class),
                           Weights => Model.Weights.Reference (Feature_Position),
                           Value   => -1.0);
                    end if;

                end;
            end loop;
        end if;

    end Update;

    function Score
      (Model    : in Model_Type;
       Features : in Features_Values_Vectors.Vector) return FLOAT_Vector_Type is
    begin

        return Scores_Distribution : FLOAT_Vector_Type
          (Index_Type (Model.First_Class) .. Index_Type (Model.Last_Class))
            := (others => 0.0) do

            for Feature_Value : Feature_Value_Type of Features loop
                if Feature_Value.Value /= 0.0 then

                    declare
                        Feature_Position : constant Weights_Maps.Cursor
                          := Model.Weights.Find (To_String (Feature_Value.Feature));

                    begin
                        if Weights_Maps.Has_Element (Feature_Position) then


                            declare
                                Weights_Item : constant Weights_Maps.Constant_Reference_Type
                                  := Model.Weights.Constant_Reference
                                    (Feature_Position);

                                Class_Position : Class_To_Value_Sparse_Vectors.Cursor;
                            begin

                                if Weights_Item.Update_Count > Model.Min_Update_Cutoff then

                                    Class_Position := Weights_Item.Classes.First;
                                    while Class_To_Value_Sparse_Vectors.Has_Element (Class_Position) loop

                                        declare
                                            Class : Index_Type renames
                                                      Weights_Item.Classes.Index
                                                        (Class_Position);

                                            Weight : Float_Type renames
                                                       Weights_Item.Element.Classes.Element
                                                         (Class_Position).Weight;
                                        begin
                                            Scores_Distribution (Class) :=
                                              Scores_Distribution (Class) +
                                              (Feature_Value.Value * Weight);
                                        end;

                                        Weights_Item.Classes.Next (Class_Position);
                                    end loop;
                                end if;
                            end;
                        end if;
                    end;

                end if;
            end loop;

        end return;

    end Score;

    function Predict
      (Model    : Model_Type;
       Features : Features_Values_Vectors.Vector)
       return Index_Type is

        Scores : constant FLOAT_Vector_Type
          := Model.Score (Features);

        Max_Class : Index_Type := Index_Type (Model.First_Class);
        Max_Value : Float_Type := Float_Type'First;
    begin

        Max_Value := Scores (Max_Class);

        for Class in Scores'Range loop
            declare
                Value : Float_Type renames Scores (Class);
            begin
                --Put_Line ("SCORE" & Class'Img & " =>" & Value'Img);

                if Value > Max_Value
                  or else (Value = Max_Value and Class > Max_Class) then

                    Max_Value := Value;
                    Max_Class := Class;
                end if;
            end;
        end loop;

        return Max_Class;
    end Predict;

    procedure Print_Stats
      (Model : in out Model_Type) is

        Sum : Integer := 0;
        Tot : Integer := 0;
    begin

        for Feature_Position in Model.Weights.Iterate loop
            declare
                Feature : constant String
                  := Weights_Maps.Key (Feature_Position);

                Weights : constant Weights_Maps.Constant_Reference_Type
                  := Model.Weights.Constant_Reference (Feature_Position);

            begin
                Put_Line (Feature & " -> " & Weights.Element.Classes.Length'Img);

                Sum := Sum + Weights.Classes.Length;
                Tot := Tot + 1;
            end;
        end loop;
        Put_Line (Integer (Sum / Tot)'Img);

    end Print_Stats;

    procedure Average_Weights
      (Model : in out Model_Type) is
    begin

        for Feature_Position in Model.Weights.Iterate loop
            declare

                Weights : constant Weights_Maps.Constant_Reference_Type
                  := Model.Weights.Constant_Reference (Feature_Position);

                New_Feature_Weights : Class_To_Value_Sparse_Vectors.Sparse_Vector_Type;

                Class_Position      : Class_To_Value_Sparse_Vectors.Cursor
                  := Weights.Element.Classes.First;
            begin
                --Put_Line("--" & Feature & "--");


                while Class_To_Value_Sparse_Vectors.Has_Element (Class_Position) loop
                    declare
                        Class : constant Index_Type
                          := Weights.Classes.Index (Class_Position);

                        Element : constant Feature_Class_Info_Type
                          := Weights.Classes.Element (Class_Position);

                        Total : constant Float_Type
                          := Element.Totals +
                            (Float_Type (Model.Instances_Count -
                                       Weights.Classes.Element (Class_Position).Timestamps)
                             * Element.Weight);

                        Averaged : Float_Type
                          := Total / Float_Type (Model.Instances_Count);
                    begin

                        --Put_Line ("W>" & Class'Img & " =>" & Total'Img);

                        -- Round 3
                        Averaged :=
                          Float_Type'Rounding (Averaged * 1_000.0) / 1_000.0;

                        if Averaged /= 0.0 then

                            New_Feature_Weights.Insert
                              (Class,
                               (Weight     => Averaged,
                                Totals     => Element.Totals,
                                Timestamps => Element.Timestamps));
                            --Put_Line ("AVG =" & Averaged'Img);
                        end if;

                    end;

                    Weights.Classes.Next (Class_Position);
                end loop;

                declare
                    procedure Update
                      (Key     : String;
                       Element : in out Update_And_Classes_Type) is
                        pragma Unreferenced (Key);
                    begin
                        Element.Classes.Clear;
                        Element.Classes := New_Feature_Weights;
                    end Update;
                begin
                    Model.Weights.Update_Element
                      (Feature_Position, Update'Access);
                    -- Cannot use Replace due to cursor tampering
                end;
            end;
        end loop;

    end Average_Weights;

    procedure Serialize
      (Model          : in Model_Type;
       Model_Filename : in String) is

        SFile : Stream_IO.File_Type;
        SAcc  : Stream_IO.Stream_Access;
    begin
        Stream_IO.Create (SFile, Stream_IO.Out_File, Model_Filename);
        SAcc := Stream_IO.Stream (SFile);

        Model_Type'Output (SAcc, Model);

        Stream_IO.Close (SFile);
    end Serialize;

    procedure Load
      (Model          : out Model_Type;
       Model_Filename : in  String) is

        SFile : Stream_IO.File_Type;
        SAcc  : Stream_IO.Stream_Access;
    begin
        Stream_IO.Open (SFile, Stream_IO.In_File, Model_Filename);
        SAcc := Stream_IO.Stream (SFile);

        Model := Model_Type'Input (SAcc);

        Stream_IO.Close (SFile);
    end Load;

    procedure Prune_Features
      (Model : in out Model_Type) is

        package String_Vectors is new
          Ada.Containers.Indefinite_Vectors
            (Index_Type   => Natural,
             Element_Type => String);

        Features_To_Delete : String_Vectors.Vector;
    begin

        Features_To_Delete.Reserve_Capacity (Model.Weights.Length);

        for Position in Model.Weights.Iterate loop
            if Weights_Maps.Element (Position).Update_Count
              <= Model.Min_Update_Cutoff then
                Features_To_Delete.Append (Weights_Maps.Key (Position));
            end if;
        end loop;

        for Feature of Features_To_Delete loop
            Model.Weights.Delete (Feature);
        end loop;

    end Prune_Features;

end MLColl.Averaged_Perceptron;
