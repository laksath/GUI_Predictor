#!/usr/bin/env python
from __future__ import print_function

from os.path import basename
import string
import random
import webbrowser
import json

#utils


class Utils:
    @staticmethod
    def get_random_text(length_text=10, space_number=1, with_upper_case=True):
        results = []
        while len(results) < length_text:
            char = random.choice(string.ascii_letters[:26])
            results.append(char)
        if with_upper_case:
            results[0] = results[0].upper()

        current_spaces = []
        while len(current_spaces) < space_number:
            space_pos = random.randint(2, length_text - 3)
            if space_pos in current_spaces:
                break
            results[space_pos] = " "
            if with_upper_case:
                results[space_pos + 1] = results[space_pos - 1].upper()

            current_spaces.append(space_pos)

        return ''.join(results)



#compiler

#Node
class Node:
    def __init__(self, key, parent_node, content_holder):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.content_holder = content_holder

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        print(self.key)
        for child in self.children:
            child.show()

    def render(self, mapping, rendering_function=None):
        content = ""
        for child in self.children:
            content += child.render(mapping, rendering_function)

        value = mapping[self.key]
        if rendering_function is not None:
            value = rendering_function(self.key, value)

        if len(self.children) != 0:
            value = value.replace(self.content_holder, content)

        return value


class Compiler:
    def __init__(self, dsl_mapping_file_path):
        with open(dsl_mapping_file_path) as data_file:
            self.dsl_mapping = json.load(data_file)

        self.opening_tag = self.dsl_mapping["opening-tag"]
        self.closing_tag = self.dsl_mapping["closing-tag"]
        self.content_holder = self.opening_tag + self.closing_tag

        self.root = Node("body", None, self.content_holder)

    def compile(self, input_file_path, output_file_path, rendering_function=None):
        dsl_file = open(input_file_path)
        current_parent = self.root

        for token in dsl_file:
            token = token.replace(" ", "").replace("\n", "")

            if token.find(self.opening_tag) != -1:
                token = token.replace(self.opening_tag, "")

                element = Node(token, current_parent, self.content_holder)
                current_parent.add_child(element)
                current_parent = element
            elif token.find(self.closing_tag) != -1:
                current_parent = current_parent.parent
            else:
                tokens = token.split(",")
                for t in tokens:
                    element = Node(t, current_parent, self.content_holder)
                    current_parent.add_child(element)

        output_html = self.root.render(self.dsl_mapping, rendering_function=rendering_function)
        with open(output_file_path, 'w') as output_file:
            output_file.write(output_html)


def web_compile(input_file):
    FILL_WITH_RANDOM_TEXT = True
    TEXT_PLACE_HOLDER = "[]"

    dsl_path = "compiler/assets/web-dsl-mapping.json"
    compiler = Compiler(dsl_path)


    def render_content_with_text(key, value):
        if FILL_WITH_RANDOM_TEXT:
            if key.find("btn") != -1:
                value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text())
            elif key.find("title") != -1:
                value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=5, space_number=0))
            elif key.find("text") != -1:
                value = value.replace(TEXT_PLACE_HOLDER,
                                    Utils.get_random_text(length_text=56, space_number=7, with_upper_case=False))
        return value

    file_uid = basename(input_file)[:basename(input_file).find(".")]
    path = input_file[:input_file.find(file_uid)]

    input_file_path = "{}{}.gui".format(path, file_uid)
    output_file_path = "{}{}.html".format(path, file_uid)
    print(output_file_path)
    compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text)
    print(output_file_path)
    url = output_file_path
    webbrowser.open(url,new=2)