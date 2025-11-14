# Code originally developed for CSE2510 Machine Learning at TU Delft
# Adopted for AI Case Study Course at TU Delft by Vasil Dakov
# by Yorick de Vries and Jordi Smit

import os
import shlex
import subprocess
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--jupyter-notebook-name',
        default='exercises_with_answers.ipynb',
        help='Enter the name of the jupyter notebook to convert (default: %(default)s)',
    )
    return parser.parse_args()

def remove_answers(notebook_file):
    if os.path.isfile(notebook_file):
        cleaned_file = notebook_file.replace('.ipynb', '_cleaned.ipynb')

        command = [
            "jupyter", "nbconvert", notebook_file,
            "--to", "notebook",
            "--ClearOutputPreprocessor.enabled=True",
            "--stdout"
        ]

        with open(cleaned_file, "w", newline='\n', encoding='utf-8') as f:
            result = subprocess.run(command, stdout=f, cwd=os.getcwd(), shell=False, check=True)
            assert result.returncode == 0, f"Process returned error code: {result}"

        student_file_path = os.path.join(
            os.getcwd(),
            notebook_file.replace('_with_answers', '_student.ipynb')
        )

        with open(cleaned_file, "r", encoding='utf-8') as answer_file, \
             open(student_file_path, "w", newline='\n', encoding="utf-8") as student_file:
            write_down = True
            dump_dummy_answer = False
            lines = answer_file.readlines()
            for i, line in enumerate(lines):
                if "END OF ANSWER" in line:
                    assert not write_down, "No corresponding START ANSWER found earlier, line: " + str(i)
                    write_down = True
                if write_down:
                    student_file.write(line)
                if not write_down and dump_dummy_answer:
                    if line.count('_') == 2:
                        my_temp_line = line.split('_')
                        my_temp_line[1] = '_Write your answer here._\\n'
                        my_new_line = ''.join(my_temp_line)
                        student_file.write(my_new_line)
                        dump_dummy_answer = False
                if "START OF ANSWER" in line:
                    assert write_down, "Previous answer did not have an END ANSWER yet, line: " + str(i)
                    write_down = False
                    if '[//]:' in line:
                        dump_dummy_answer = True
            assert write_down, "The last START ANSWER is never closed with END ANSWER, line"
        os.remove(cleaned_file)

if __name__ == '__main__':
    args = get_args()
    remove_answers(args.jupyter_notebook_name)
