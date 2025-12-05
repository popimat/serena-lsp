"""
Language server-related tools
"""

import dataclasses
import os
from collections.abc import Sequence
from copy import copy
from typing import Any

from serena.tools import (
    SUCCESS_RESULT,
    Tool,
    ToolMarkerSymbolicEdit,
    ToolMarkerSymbolicRead,
)
from serena.tools.tools_base import ToolMarkerOptional
from solidlsp.ls_types import SymbolKind


def _format_range(rng: dict[str, Any]) -> str:
    start = rng["start"]
    end = rng["end"]
    return f"{start['line'] + 1}:{start['character'] + 1}-{end['line'] + 1}:{end['character'] + 1}"


def _sanitize_symbol_dict(symbol_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize a symbol dictionary inplace by removing unnecessary information.
    """
    # We replace the location entry, which repeats line information already included in body_location
    # and has unnecessary information on column, by just the relative path.
    symbol_dict = copy(symbol_dict)
    s_relative_path = symbol_dict.get("location", {}).get("relative_path")
    if s_relative_path is not None:
        symbol_dict["relative_path"] = s_relative_path
    symbol_dict.pop("location", None)

    rng = symbol_dict.get("range")
    if rng:
        symbol_dict["range"] = _format_range(rng)

    # also remove name, name_path should be enough
    symbol_dict.pop("name")
    return symbol_dict


def _clean_none_and_empty(obj: Any) -> Any:
    """
    Recursively remove None values and empty lists/dicts from a dictionary or list.
    """
    if isinstance(obj, dict):
        return {
            k: v
            for k, v in ((k, _clean_none_and_empty(v)) for k, v in obj.items())
            if v is not None and (not isinstance(v, (list, dict)) or len(v) > 0)
        }
    if isinstance(obj, list):
        return [
            v
            for v in (_clean_none_and_empty(v) for v in obj)
            if v is not None and (not isinstance(v, (list, dict)) or len(v) > 0)
        ]
    return obj


class _LanguageServerToolMixin:
    """Helper mixin for tools that need direct language server access."""

    def _get_language_server(self, relative_path: str):
        if not self.agent.is_using_language_server():
            raise RuntimeError("Cannot create LanguageServer; agent is not in language server mode.")
        language_server_manager = self.agent.get_language_server_manager_or_raise()
        return language_server_manager.get_language_server(relative_path)


class RestartLanguageServerTool(Tool, ToolMarkerOptional):
    """Restarts the language server, may be necessary when edits not through Serena happen."""

    def apply(self) -> str:
        """Use this tool only on explicit user request or after confirmation.
        It may be necessary to restart the language server if it hangs.
        """
        self.agent.reset_language_server_manager()
        return SUCCESS_RESULT


class GetSymbolsOverviewTool(Tool, ToolMarkerSymbolicRead):
    """
    Gets an overview of the top-level symbols defined in a given file.
    """

    def apply(self, relative_path: str, max_answer_chars: int = -1) -> str:
        """
        Use this tool to get a high-level understanding of the code symbols in a file.
        This should be the first tool to call when you want to understand a new file, unless you already know
        what you are looking for.

        :param relative_path: the relative path to the file to get the overview of
        :param max_answer_chars: if the overview is longer than this number of characters,
            no content will be returned. -1 means the default value from the config will be used.
            Don't adjust unless there is really no other way to get the content required for the task.
        :return: a serialized object (YAML/JSON) containing info about top-level symbols in the file
        """
        symbol_retriever = self.create_language_server_symbol_retriever()
        file_path = os.path.join(self.project.project_root, relative_path)

        # The symbol overview is capable of working with both files and directories,
        # but we want to ensure that the user provides a file path.
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File or directory {relative_path} does not exist in the project.")
        if os.path.isdir(file_path):
            raise ValueError(f"Expected a file path, but got a directory path: {relative_path}. ")
        result = symbol_retriever.get_symbol_overview(relative_path)[relative_path]
        result_str = self._to_output([dataclasses.asdict(i) for i in result])
        return self._limit_length(result_str, max_answer_chars)


class FindSymbolTool(Tool, ToolMarkerSymbolicRead):
    """
    Performs a global (or local) search using the language server backend.
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path_pattern: str,
        depth: int = 0,
        relative_path: str = "",
        include_body: bool = False,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        substring_matching: bool = False,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Retrieves information on all symbols/code entities (classes, methods, etc.) based on the given name path pattern.
        The returned symbol information can be used for edits or further queries.
        Specify `depth > 0` to also retrieve children/descendants (e.g., methods of a class).

        A name path is a path in the symbol tree *within a source file*.
        For example, the method `my_method` defined in class `MyClass` would have the name path `MyClass/my_method`.
        If a symbol is overloaded (e.g., in Java), a 0-based index is appended (e.g. "MyClass/my_method[0]") to
        uniquely identify it.

        To search for a symbol, you provide a name path pattern that is used to match against name paths.
        It can be
         * a simple name (e.g. "method"), which will match any symbol with that name
         * a relative path like "class/method", which will match any symbol with that name path suffix
         * an absolute name path "/class/method" (absolute name path), which requires an exact match of the full name path within the source file.
        Append an index `[i]` to match a specific overload only, e.g. "MyClass/my_method[1]".

        :param name_path_pattern: the name path matching pattern (see above)
        :param depth: depth up to which descendants shall be retrieved (e.g. use 1 to also retrieve immediate children;
            for the case where the symbol is a class, this will return its methods).
            Default 0.
        :param relative_path: Optional. Restrict search to this file or directory. If None, searches entire codebase.
            If a directory is passed, the search will be restricted to the files in that directory.
            If a file is passed, the search will be restricted to that file.
            If you have some knowledge about the codebase, you should use this parameter, as it will significantly
            speed up the search as well as reduce the number of results.
        :param include_body: If True, include the symbol's source code. Use judiciously.
        :param include_kinds: Optional. List of LSP symbol kind integers to include. (e.g., 5 for Class, 12 for Function).
            Valid kinds: 1=file, 2=module, 3=namespace, 4=package, 5=class, 6=method, 7=property, 8=field, 9=constructor, 10=enum,
            11=interface, 12=function, 13=variable, 14=constant, 15=string, 16=number, 17=boolean, 18=array, 19=object,
            20=key, 21=null, 22=enum member, 23=struct, 24=event, 25=operator, 26=type parameter.
            If not provided, all kinds are included.
        :param exclude_kinds: Optional. List of LSP symbol kind integers to exclude. Takes precedence over `include_kinds`.
            If not provided, no kinds are excluded.
        :param substring_matching: If True, use substring matching for the last element of the pattern, such that
            "Foo/get" would match "Foo/getValue" and "Foo/getData".
        :param max_answer_chars: Max characters for the result. If exceeded, no content is returned.
            -1 means the default value from the config will be used.
        :return: a list of symbols (with locations) matching the name.
        """
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()
        symbols = symbol_retriever.find_by_name(
            name_path_pattern,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
            substring_matching=substring_matching,
            within_relative_path=relative_path,
        )
        symbol_dicts = [_sanitize_symbol_dict(s.to_dict(kind=True, location=True, depth=depth, include_body=include_body)) for s in symbols]
        result = self._to_output(symbol_dicts)
        return self._limit_length(result, max_answer_chars)


class FindReferencingSymbolsTool(Tool, ToolMarkerSymbolicRead):
    """
    Finds symbols that reference the given symbol using the language server backend
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path: str,
        relative_path: str,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds references to the symbol at the given `name_path`. The result will contain metadata about the referencing symbols
        as well as a short code snippet around the reference.

        :param name_path: for finding the symbol to find references for, same logic as in the `find_symbol` tool.
        :param relative_path: the relative path to the file containing the symbol for which to find references.
            Note that here you can't pass a directory but must pass a file.
        :param include_kinds: same as in the `find_symbol` tool.
        :param exclude_kinds: same as in the `find_symbol` tool.
        :param max_answer_chars: same as in the `find_symbol` tool.
        :return: a list of serialized objects (YAML/JSON) with the symbols referencing the requested symbol
        """
        include_body = False  # It is probably never a good idea to include the body of the referencing symbols
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()
        references_in_symbols = symbol_retriever.find_referencing_symbols(
            name_path,
            relative_file_path=relative_path,
            include_body=include_body,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
        )
        reference_dicts = []
        for ref in references_in_symbols:
            ref_dict = ref.symbol.to_dict(kind=True, location=True, depth=0, include_body=include_body)
            ref_dict = _sanitize_symbol_dict(ref_dict)
            if not include_body:
                ref_relative_path = ref.symbol.location.relative_path
                assert ref_relative_path is not None, f"Referencing symbol {ref.symbol.name} has no relative path, this is likely a bug."
                content_around_ref = self.project.retrieve_content_around_line(
                    relative_file_path=ref_relative_path, line=ref.line, context_lines_before=1, context_lines_after=1
                )
                ref_dict["content_around_reference"] = content_around_ref.to_display_string()
            reference_dicts.append(ref_dict)
        result = self._to_output(reference_dicts)
        return self._limit_length(result, max_answer_chars)


class ReplaceSymbolBodyTool(Tool, ToolMarkerSymbolicEdit):
    """
    Replaces the full definition of a symbol using the language server backend.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        r"""
        Replaces the body of the symbol with the given `name_path`.

        The tool shall be used to replace symbol bodies that have been previously retrieved
        (e.g. via `find_symbol`).
        IMPORTANT: Do not use this tool if you do not know what exactly constitutes the body of the symbol.

        :param name_path: for finding the symbol to replace, same logic as in the `find_symbol` tool.
        :param relative_path: the relative path to the file containing the symbol
        :param body: the new symbol body. The symbol body is the definition of a symbol
            in the programming language, including e.g. the signature line for functions.
            IMPORTANT: The body does NOT include any preceding docstrings/comments or imports, in particular.
        """
        code_editor = self.create_code_editor()
        code_editor.replace_body(
            name_path,
            relative_file_path=relative_path,
            body=body,
        )
        return SUCCESS_RESULT


class InsertAfterSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Inserts content after the end of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Inserts the given body/content after the end of the definition of the given symbol (via the symbol's location).
        A typical use case is to insert a new class, function, method, field or variable assignment.

        :param name_path: name path of the symbol after which to insert content (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted. The inserted code shall begin with the next line after
            the symbol.
        """
        code_editor = self.create_code_editor()
        code_editor.insert_after_symbol(name_path, relative_file_path=relative_path, body=body)
        return SUCCESS_RESULT


class InsertBeforeSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Inserts content before the beginning of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Inserts the given content before the beginning of the definition of the given symbol (via the symbol's location).
        A typical use case is to insert a new class, function, method, field or variable assignment; or
        a new import statement before the first symbol in the file.

        :param name_path: name path of the symbol before which to insert content (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted before the line in which the referenced symbol is defined
        """
        code_editor = self.create_code_editor()
        code_editor.insert_before_symbol(name_path, relative_file_path=relative_path, body=body)
        return SUCCESS_RESULT


class RenameSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Renames a symbol throughout the codebase using language server refactoring capabilities.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        new_name: str,
    ) -> str:
        """
        Renames the symbol with the given `name_path` to `new_name` throughout the entire codebase.
        Note: for languages with method overloading, like Java, name_path may have to include a method's
        signature to uniquely identify a method.

        :param name_path: name path of the symbol to rename (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol to rename
        :param new_name: the new name for the symbol
        :return: result summary indicating success or failure
        """
        code_editor = self.create_code_editor()
        status_message = code_editor.rename_symbol(name_path, relative_file_path=relative_path, new_name=new_name)
        return status_message


class GetHoverInfoTool(_LanguageServerToolMixin, Tool, ToolMarkerSymbolicRead):
    """
    Gets hover information for a symbol at a specific location in a file.
    """

    def apply(self, relative_path: str, line: int, character: int) -> str:
        """
        Retrieves hover information (documentation, type info, etc.) for the code at the specified position.
        This is equivalent to hovering over the code in an IDE.

        :param relative_path: the relative path to the file
        :param line: the line number (0-based)
        :param character: the character offset (0-based)
        :return: the hover information as a string
        """
        ls = self._get_language_server(relative_path)
        result = ls.request_hover(relative_path, line, character)
        if result is None:
            return "No hover information available."

        # Format the result
        contents = result.get("contents")
        if not contents:
            return "No content in hover result."

        if isinstance(contents, str):
            return contents
        elif isinstance(contents, dict):
            return contents.get("value", str(contents))
        elif isinstance(contents, list):
            # List of MarkedString or string
            parts = []
            for part in contents:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("value", str(part)))
            return "\n\n".join(parts)

        return str(contents)


def _format_definition_entries(definitions: list[dict[str, Any]], project_root: str, project) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for definition in definitions:
        definition_path = definition.get("relativePath")
        absolute_definition_path = definition.get("absolutePath")

        if definition_path is None and absolute_definition_path is not None:
            definition_path = os.path.relpath(absolute_definition_path, project_root)

        snippet = None
        if definition_path is not None:
            display = project.retrieve_content_around_line(
                relative_file_path=definition_path,
                line=definition["range"]["start"]["line"],
                context_lines_before=2,
                context_lines_after=2,
            )
            snippet = display.to_display_string()

        rng = definition.get("range")
        formatted_range = _format_range(rng) if rng else None

        entry = {
            "relative_path": definition_path,
            "range": formatted_range,
            "snippet": snippet,
        }
        # Only include uri if we couldn't determine a relative path
        if definition_path is None:
            entry["uri"] = definition.get("uri")
            
        entries.append(entry)

    return entries


class GetDefinitionLocationsTool(_LanguageServerToolMixin, Tool, ToolMarkerSymbolicRead):
    """Retrieves the definition locations for the symbol at a given file position."""

    def apply(self, relative_path: str, line: int, character: int, max_answer_chars: int = -1) -> str:
        """
        Retrieves the definition locations for the symbol at the given file position.

        :param relative_path: the relative path to the file
        :param line: 0-indexed line number
        :param character: 0-indexed character position
        :param max_answer_chars: Optional maximum number of characters in the answer. Default uses the configured limit.
        :return: A list of definition locations with file paths and ranges.
        """
        ls = self._get_language_server(relative_path)
        definitions = ls.request_definition(relative_path, line, character)
        definition_entries = _format_definition_entries(definitions, self.project.project_root, self.project)

        payload = self._to_output(_clean_none_and_empty(definition_entries))
        return self._limit_length(payload, max_answer_chars)


def _format_reference_entries(references: list[dict[str, Any]], project_root: str, project) -> list[dict[str, Any]]:
    """Format LSP reference locations into a structured list with snippets."""
    entries: list[dict[str, Any]] = []
    for ref in references:
        ref_path = ref.get("relativePath")
        absolute_ref_path = ref.get("absolutePath")

        if ref_path is None and absolute_ref_path is not None:
            ref_path = os.path.relpath(absolute_ref_path, project_root)

        snippet = None
        if ref_path is not None:
            rng = ref.get("range")
            if rng:
                display = project.retrieve_content_around_line(
                    relative_file_path=ref_path,
                    line=rng["start"]["line"],
                    context_lines_before=2,
                    context_lines_after=2,
                )
                snippet = display.to_display_string()

        rng = ref.get("range")
        formatted_range = _format_range(rng) if rng else None

        entry = {
            "relative_path": ref_path,
            "range": formatted_range,
            "snippet": snippet,
        }
        # Only include uri if we couldn't determine a relative path
        if ref_path is None:
            entry["uri"] = ref.get("uri")

        entries.append(entry)

    return entries


class GetReferenceLocationsTool(_LanguageServerToolMixin, Tool, ToolMarkerSymbolicRead):
    """Retrieves all reference locations for the symbol at a given file position."""

    def apply(self, relative_path: str, line: int, character: int, max_answer_chars: int = -1) -> str:
        """
        Retrieves all locations where the symbol at the given position is referenced.

        :param relative_path: the relative path to the file
        :param line: 0-indexed line number
        :param character: 0-indexed character position
        :param max_answer_chars: Optional maximum number of characters in the answer. Default uses the configured limit.
        :return: A list of reference locations with file paths, ranges, and code snippets.
        """
        ls = self._get_language_server(relative_path)
        references = ls.request_references(relative_path, line, character)
        reference_entries = _format_reference_entries(references, self.project.project_root, self.project)

        payload = self._to_output(_clean_none_and_empty(reference_entries))
        return self._limit_length(payload, max_answer_chars)


def _serialize_symbol(symbol: dict[str, Any]) -> dict[str, Any]:
    serialized = copy(symbol)
    # Remove fields that are not useful for the LLM
    serialized.pop("parent", None)
    serialized.pop("glyph", None)
    serialized.pop("selectionRange", None)

    kind_value = serialized.get("kind")
    if isinstance(kind_value, SymbolKind):
        serialized["kind"] = kind_value.name
    elif isinstance(kind_value, int):
        try:
            serialized["kind"] = SymbolKind(kind_value).name
        except ValueError:
            serialized["kind"] = str(kind_value)

    children = serialized.get("children")
    if isinstance(children, list):
        serialized["children"] = [_serialize_symbol(child) for child in children]

    rng = serialized.get("range")
    if rng:
        serialized["range"] = _format_range(rng)

    location = serialized.get("location")
    if location is not None:
        loc_dict = dict(location)
        path = loc_dict.get("relativePath")
        if not path:
            path = loc_dict.get("uri")

        loc_rng = loc_dict.get("range")
        if path and loc_rng:
            serialized["location"] = f"{path}:{_format_range(loc_rng)}"
        else:
            # Fallback if something is missing
            if loc_dict.get("relativePath"):
                loc_dict.pop("uri", None)
                loc_dict.pop("absolutePath", None)
            serialized["location"] = loc_dict

    body = serialized.get("body")
    if isinstance(body, str):
        import textwrap

        lines = body.splitlines()
        if lines:
            # The first line is often the declaration and might have different indentation
            # (or be stripped already) compared to the body block.
            # We strip the first line and dedent the rest.
            first_line = lines[0].lstrip()
            if len(lines) > 1:
                # Remove common indentation from the rest
                rest = textwrap.dedent("\n".join(lines[1:]))
                serialized["body"] = first_line + "\n" + rest
            else:
                serialized["body"] = first_line

    return serialized


class GetDefiningSymbolTool(_LanguageServerToolMixin, Tool, ToolMarkerSymbolicRead):
    """Retrieves the defining symbol (with optional body) for the code at a given location."""

    def apply(self, relative_path: str, line: int, character: int, include_body: bool = False, max_answer_chars: int = -1) -> str:
        """
        Retrieves the defining symbol for the code at the given file position.

        :param relative_path: the relative path to the file
        :param line: 0-indexed line number
        :param character: 0-indexed character position
        :param include_body: whether to include the symbol's body/implementation
        :param max_answer_chars: Optional maximum number of characters in the answer. Default uses the configured limit.
        :return: Symbol information including location and optionally the body.
        """
        ls = self._get_language_server(relative_path)
        symbol = ls.request_defining_symbol(relative_path, line, character, include_body=include_body)
        definitions = ls.request_definition(relative_path, line, character)
        definition_entries = _format_definition_entries(definitions, self.project.project_root, self.project)

        payload: dict[str, Any] = {
            "symbol": _serialize_symbol(symbol) if symbol is not None else None,
            "definitions": definition_entries,
        }
        if symbol is None:
            payload["error"] = "No defining symbol found."

        output_payload = self._to_output(_clean_none_and_empty(payload))
        return self._limit_length(output_payload, max_answer_chars)
