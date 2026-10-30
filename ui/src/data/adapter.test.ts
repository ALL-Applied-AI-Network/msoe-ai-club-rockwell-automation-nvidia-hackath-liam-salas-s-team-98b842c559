import fixture from "../../../ui-data/example.json";
import { describe, expect, it } from "vitest";

import { DemoDataValidationError, parseDemoData } from "./adapter";

function copyFixture(): Record<string, any> {
  return structuredClone(fixture);
}

describe("parseDemoData", () => {
  it("accepts the documented synthetic twin fixture", () => {
    const data = parseDemoData(fixture);

    expect(data.dataset_id).toBe("phase1-synthetic-demo");
  });

  it("rejects duplicate vial identifiers", () => {
    const input = copyFixture();
    input.vials.push(structuredClone(input.vials[0]));

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError('Duplicate vial_id "vial-001" in vials')
    );
  });

  it("rejects dangling evidence references", () => {
    const input = copyFixture();
    input.vials[0].decision.evidence_refs = ["evidence-missing"];

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Decision "decision-001" references missing evidence "evidence-missing"'
      )
    );
  });

  it("rejects a top-down observation linked to another decision", () => {
    const input = copyFixture();
    input.top_down.observations[0].decision_id = "decision-other";

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Top-down observation for "vial-001" links decision "decision-other"; expected "decision-001"'
      )
    );
  });

  it("rejects a normalized bounding box that extends past the image", () => {
    const input = copyFixture();
    input.top_down.observations[0].bbox.x = 0.95;

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Bounding box for "vial-001" extends outside normalized image bounds'
      )
    );
  });

  it.each([
    {
      name: "top-down vial_id",
      mutate(input: Record<string, any>) {
        input.top_down.observations.push(
          structuredClone(input.top_down.observations[0])
        );
      },
      message: 'Duplicate vial_id "vial-001" in top_down.observations'
    },
    {
      name: "evidence_id",
      mutate(input: Record<string, any>) {
        input.evidence.push(structuredClone(input.evidence[0]));
      },
      message: 'Duplicate evidence_id "evidence-001" in evidence'
    }
  ])("rejects duplicate $name values", ({ mutate, message }) => {
    const input = copyFixture();
    mutate(input);

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(message)
    );
  });

  it("rejects a top-down observation for an unknown vial", () => {
    const input = copyFixture();
    input.top_down.observations[0].vial_id = "vial-missing";

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Top-down observation references unknown vial_id "vial-missing"'
      )
    );
  });

  it("rejects evidence assigned to a different vial", () => {
    const input = copyFixture();
    input.evidence[0].vial_id = "vial-other";

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Decision "decision-001" for "vial-001" references evidence "evidence-001" owned by "vial-other"'
      )
    );
  });

  it("rejects an evidence record for an unknown vial", () => {
    const input = copyFixture();
    const orphan = structuredClone(input.evidence[0]);
    orphan.evidence_id = "evidence-orphan";
    orphan.vial_id = "vial-missing";
    input.evidence.push(orphan);

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Evidence "evidence-orphan" references unknown vial_id "vial-missing"'
      )
    );
  });

  it("rejects a normalized position outside its vial bounding box", () => {
    const input = copyFixture();
    input.top_down.observations[0].position.x = 0.9;

    expect(() => parseDemoData(input)).toThrowError(
      new DemoDataValidationError(
        'Position for "vial-001" lies outside its normalized bounding box'
      )
    );
  });

  it.each([
    {
      name: "decision outcome",
      mutate(input: Record<string, any>) {
        input.vials[0].decision.outcome = "rework";
      }
    },
    {
      name: "explanation source",
      mutate(input: Record<string, any>) {
        input.vials[0].decision.explanation.source = "free_text_model";
      }
    },
    {
      name: "provenance kind",
      mutate(input: Record<string, any>) {
        input.vials[0].decision.delta_e.provenance.kind = "assumed";
      }
    }
  ])("rejects an unsupported $name", ({ mutate }) => {
    const input = copyFixture();
    mutate(input);

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it("rejects evidence-bearing numbers without provenance", () => {
    const input = copyFixture();
    delete input.top_down.observations[0].bbox.provenance;

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it("rejects zero-area normalized geometry", () => {
    const input = copyFixture();
    input.top_down.observations[0].bbox.width = 0;

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it("rejects data not labelled as synthetic twin data", () => {
    const input = copyFixture();
    input.label = "Operator demo data";

    expect(() => parseDemoData(input)).toThrowError(DemoDataValidationError);
  });

  it.each(["keep", "discard", "insufficient_evidence"])(
    "accepts the supported %s decision",
    (outcome) => {
      const input = copyFixture();
      input.vials[0].decision.outcome = outcome;

      expect(parseDemoData(input).vials[0].decision.outcome).toBe(outcome);
    }
  );

  it.each(["deterministic_template", "cosmos_reason2"])(
    "accepts the supported %s explanation source",
    (source) => {
      const input = copyFixture();
      input.vials[0].decision.explanation.source = source;

      expect(parseDemoData(input).vials[0].decision.explanation.source).toBe(
        source
      );
    }
  );
});
